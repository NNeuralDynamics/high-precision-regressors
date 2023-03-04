#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import dataframe
from pyspark.sql.types import StructType, StructField, DoubleType
import sklearn.model_selection as ms
from sklearn import metrics
import time
import sys
import logging
import uuid
import os
import shutil
import json
import argparse

SEED = 42


class TerminateOnBaseline(tf.keras.callbacks.Callback):
    """ Callback that terminates training when monitored value reaches a specified baseline
        or has not improved for a while
    """
    def __init__(self, config):
        """  Member function for init
            arguments:
                config: the configuration file
        """
        super(TerminateOnBaseline, self).__init__()
        self.monitor = config['monitor']
        self.baseline = np.Inf
        self.patience = config['patience']
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        self.best_weights = None
        self.best_epoch = 0
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        """ member function to define what is evaluated at the end of an epoch
            arguments:
                epoch: the epoch
                logs: history log
        """
        logs = logs or {}
        value = logs.get(self.monitor)
        if epoch == 0:
            self.baseline = value/10.
        if np.less(value, self.best):
            self.best = value
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
        else:
            self.wait += 1
        if value is not None:
            if value <= self.baseline and self.wait >= self.patience:
                self.stopped_epoch = epoch
                logging.info('\nepoch %d: Reached baseline, terminating training with patience lost' % epoch)
                self.model.stop_training = True
                logging.info('Restoring model weights from the end of the best epoch: ' + str(self.best_epoch + 1))
                logging.info('Value of monitored metric: ' + self.monitor + ' = ' + str(self.best))
                self.config['best_epoch'] = self.best_epoch + 1
                self.model.set_weights(self.best_weights)
            elif self.wait >= self.patience:
                self.baseline *= 2.5
                self.wait = self.patience/2


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """ Learning rate scheduler which sets the learning rate according to schedule.
        arguments:
            schedule: a function that takes an epoch index (integer, indexed from 0) and current learning rate
                      as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        print("learning rare scheduler initialized")

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
            
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        logging.info(" Epoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


def lr_schedule(epoch, lr):
    """ Helper function to retrieve the scheduled learning rate based on epoch.
        arguments:
            epoch: the current epoch number
            lr: the current learning rate
        returns:
            lr: the adjusted learning rate
    """
    LR_SCHEDULE = [
        # (epoch to start, learning rate) tuples
        (10, 0.001),
        (75, 0.0007),
        (100, 0.0003),
        (125, 0.0001),
        (150, 0.00007),
        (175, 0.00003),
        (200, 0.00001),
        (225, 0.000007),
        (250, 0.000003),
        (275, 0.000001)
    ]
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


def lr_schedule_polynomial(starter_learning_rate=0.001, end_learning_rate=0.0005, decay_steps=100000, power=0.5):
    """ Plynomial learning schedule
        arguments:
            starter_learning_rate: a float for the starting learning rate
            end_learning_rate: a float for the ending leadning rate
            decay_steps: an integer for the number of decay steps
            power: a float for the power of the decay rate
    """
    lr_schedule_poly = tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=power)
    
    return lr_schedule_poly


def lr_schedule_exponential(starter_learning_rate=0.001, decay_steps=100000, decay_rate=0.5):
    """ Plynomial learning schedule a*exp(k*t/tau) for t steps
        arguments:
            starter_learning_rate: a float for the starting learning rate (a)
            decay_steps: an integer for the number of decay steps (tau)
            decay_rate: the decay rate of the exponential (tau)
    """
    lr_schedule_exp = tf.keras.optimizers.schedules.ExponentialDecay(
        starter_learning_rate,
        decay_steps,
        decay_rate)
    
    return lr_schedule_exp


def timediff(x):
    """ a function to convert seconds to hh:mm:ss
        argument:
            x: time in seconds
        returns:
            time in hh:mm:ss
    """
    return "{}:{}:{}".format(int(x/3600), str(int(x/60%60)).zfill(2), str(round(x - int(x/3600)*3600 - int(x/60%60)*60)).zfill(2))


def block_skip_net(x, width, activation='relu', squeeze=False):
    """ the basic building block of a dnn with skip connections
        arguments:
            x: the input
            width: the width of the hidden layers
            activation: the activation function: 'relu', 'elu', 'swish' (silu), 'leaky_relu', 'softplus'
            squeeze: a boolean specifying wheher the skip units are squeezed
        returns:
            res: the skip net block
    """
    # layer 1 with non-linearity
    y = tf.keras.layers.Dense(width, activation=activation)(x)
    # layer 2 with non-linearity
    y = tf.keras.layers.Dense(width, activation=activation)(y)

    # layer 3 with short circuit
    if squeeze:
        y = tf.keras.layers.Dense(x.shape[1], activation='linear')(y)
    else:
        y = tf.keras.layers.Dense(width, activation='linear')(y)
    if x.shape[1] != y.shape[1]:
        x_reshape = tf.keras.layers.Dense(width, activation='linear', use_bias=False, trainable = True)(x)
        res = tf.keras.layers.Add()([x_reshape,y])
    else:
        res = tf.keras.layers.Add()([x,y]) # check syntax

    res = tf.keras.layers.Activation(activation)(res)
     
    return res


def nets(config):
    """ the tensorflow model builder
        arguments:
            config: the configuration file
        returns:
            regressor: the tensorflow model
    """
    # set to float64
    tf.keras.backend.set_floatx('float64')
    
    
    # define the tensorflow model
    if config["model_type"] == 'dnn':
        x = tf.keras.layers.Input(shape=(config["input_shape"],))
        y = tf.keras.layers.Dense(config['width'], activation=config['activation'])(x)
        for i in range(1, config['depth']):
            y = tf.keras.layers.Dense(config['width'], activation=config['activation'])(y)
        y = tf.keras.layers.Dense(1, activation='linear')(y)
        regressor = tf.keras.models.Model(x, y)
        
    elif config["model_type"] in ['skip', 'squeeze']:
        x = tf.keras.layers.Input(shape=(config["input_shape"],))
        y = block_skip_net(x, config['width'], activation=config['activation'], squeeze=config["model_type"]=='squeeze')
        for i in range(1, config['depth']):
            y = block_skip_net(y, config['width'], activation=config['activation'], squeeze=config["model_type"]=='squeeze')
        y = tf.keras.layers.Dense(1, activation='linear')(y)
        regressor = tf.keras.models.Model(x, y)
        
    else:
        logging.error(' '+config["model_type"]+' not implemented. model_type can be either dnn, skip or squeeze')
        
        
    # save parameter counts
    config["trainable_parameters"] = int(np.sum([np.prod(v.get_shape()) for v in regressor.trainable_weights]))
    config["non_trainable_parameters"] = int(np.sum([np.prod(v.get_shape()) for v in regressor.non_trainable_weights]))
    config["total_parameters"] = int(config["trainable_parameters"] + config["non_trainable_parameters"])
    
    # save config
    with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    return regressor

def normalize(df, config, var_y=None):
    """ a function to normaliza the target distribution
        arguments:
            df: dataframe containing the target variable
            config: the config file with the run configuration
            var_y: the variable to be normalized
    """
    if not var_y:
        var_y = config['var_y']
    config["mu"] = df[var_y].mean()
    config["sigma"] = df[var_y].std()
    df['y_scaled'] = (df[var_y] - config["mu"])/config["sigma"]

    
def runML(df, config):
    """ run the DNN
        arguments:
            df: the dataframe including, training, validation and test
            config: the configuration dictionary for the hyperparameters
    """
    
    # define variables and target
    var_x = ['x'+str(i+1) for i in range(config['input_shape'])]
    var_y = config['var_y']
    X = df[var_x]
    
    if config['scaled'] == 'normal':
        normalize(df, config)
        y = df['y_scaled'].values
    elif config['scaled'] == 'log':
        df['y_log_scaled'] = np.log(df[var_y])
        normalize(df, config, var_y='y_log_scaled')
        y = df['y_scaled'].values
    else:
        y = df[var_y].values
        config["mu"] = 0
        config["sigma"] = 1
    
    # Split for training and testing
    x_train, x_test, y_train, y_test = ms.train_test_split(X.values, y, test_size=0.2, random_state=SEED)
    eval_set = [(x_train, y_train), (x_test, y_test)]
    
    regressor = nets(config)

    # print the summary
    regressor.summary()
    
    # define the model checkpoint callback
    checkpoint_filepath = config['directory']+'/checkpoint-'+config['model-uuid']+'-'+config['monitor']+'.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, 
                                                                   save_weights_only=True, 
                                                                   monitor=config['monitor'], 
                                                                   mode='min', 
                                                                   save_best_only=True)
    
    # compile the regressor
    if config['lr_decay_type'] == 'exp':
        lr_schedule = lr_schedule_exponential(
            starter_learning_rate=config['initial_lr'],
            decay_steps=config['decay_steps'])
    elif config['lr_decay_type'] == 'poly':
        lr_schedule  = lr_schedule_polynomial(
            starter_learning_rate=config['initial_lr'], 
            end_learning_rate=config['final_lr'], 
            decay_steps=config['decay_steps']),
    elif config['lr_decay_type'] == 'const':
        lr_schedule = 0.001
    else:
        raise ValueError('lr type not defined. Has to be exp or poly')
    
    regressor.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr_schedule),
        loss=config['loss'],
        metrics=config['metrics'])
    
    # run the regressor
    start = time.time()
    history = regressor.fit(
                x_train, y_train,
                epochs=config['epochs'],
                verbose=config['verbose'],
                validation_split=config['validation_split'],
                batch_size=config['batch_size'],
                steps_per_epoch=config['steps_per_epoch'],
                callbacks=[TerminateOnBaseline(config=config), 
                           model_checkpoint_callback,
                           # CustomLearningRateScheduler(lr_scheduler)
                          ]
    )
    config["fit_time"] = timediff(time.time() - start)
    
    return x_test, y_test, regressor, history


def plot_loss(history, dir_name):
    """ plotting routine
        argument:
            history: the tf history object
    """
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('y')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/training-evaluation.pdf', dpi=300)

    
def load_data(config):
    """ Load the data using pyspark and return a pandas dataframe
    """
    
    # Spark session and configuration
    logging.info(' creating Spark session')
    spark = (SparkSession.builder.master("local[48]")
             .config('spark.executor.instances', 16)
             .config('spark.executor.cores', 16)
             .config('spark.executor.memory', '10g')
             .config('spark.driver.memory', '15g')
             .config('spark.memory.offHeap.enabled', True)
             .config('spark.memory.offHeap.size', '20g')
             .config('spark.dirver.maxResultSize', '20g')
             .config('spark.debug.maxToStringFields', 100)
             .appName("amp.hell").getOrCreate())

    # Enable Arrow-based columnar data 
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set(
        "spark.sql.execution.arrow.pyspark.fallback.enabled", "true"
    )
    logging.info(' Spark initialized')
    
    # read the data into a spark frame
    start = time.time()
    path = config['data_path']
    if path[-1] != '/':
        path = path + '/'
    header = ['x'+str(i+1) for i in range(config['input_shape'])] + ['yN', 'y_2'] 
    schema = StructType([StructField(header[i], DoubleType(), True) for i in range(config['input_shape']+2)])
    folded = 'F' if config['folded'] else ''
    df = spark.read.options(delimiter=',').schema(schema).format("csv").load(path+str(config['input_shape'])+'D'+folded+'/train/*.csv.*', header='true')

    logging.info(' data loaded into Spark session in {:.3f} seconds'.format(time.time() - start))
    
    # transfer the data to a pandas dataframe
    start = time.time()
    df_p = df.limit(config['sample-size']).toPandas() 

    logging.info(' data loaded into pandas dataframe in {:.3f} seconds'.format(time.time() - start))
    
    return df_p, spark


def post_process(regressor, x_test, y_test, history, config):
    """ post process the regressor to check for accuracy and save everything
        argumants:
            regressor: the tensorflow regressor object
            history: the history object for the training
            config: the configuration for the training
    """
    # check accuracy
    logging.info(' running the DNN predictions and accuracy computation')
    y_pred = regressor.predict(x_test, verbose=config['verbose'])
    abs_score = (1 - np.mean(np.abs((y_pred.T - y_test)/y_test)))*100
    r2_score = metrics.r2_score(y_test, y_pred)*100
    config["abs_score"] = abs_score
    config["r2_score"] = r2_score
    logging.info(' relative accuracy: {:.6f}%  |---|  R2 score: {:.6f}%'.format(abs_score, r2_score))

    # save the regressor
    logging.info(' saving the net')
    regressor.load_weights(config['directory']+'/checkpoint-'+config['model-uuid']+'-'+config['monitor']+'.hdf5')
    regressor.save(config['directory']+'/dnn-'+str(config['depth'])+'-'+str(config['width'])+'-'+config['activation']+'-'+str(config['batch_size'])+'-adam-'+config['lr_decay_type']+'-schedule-'+config['loss']+'-'+config['monitor']+f'-{abs_score:.6f}-{r2_score:.6f}.tfm.hdf5')
        
    #plot the training history
    logging.info(' printing training evaluation plots')
    plot_loss(history, config['directory'])
    
    # end time
    config["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
                 
    # save config
    with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # remove preliminary config file
    os.remove(config['directory']+'/config-'+config['model-uuid']+'-prelim.json')
        
    # move directory
    shutil.move(config['directory'], config['directory']+'-'+config['scaled']+'-'+str(config['depth'])+'-'+str(config['width'])+'-'+config['activation']+'-'+str(config['batch_size'])+'-adam-'+config['lr_decay_type']+'-schedule-'+config['loss']+'-'+config['monitor']+f'-{abs_score:.6f}-{r2_score:.6f}')    
    
    
def main():
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="A python script implementing DNN and sk-DNN for high-precision machine learning",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", help="configuration file for the run")
    args = vars(parser.parse_args())
    
    # set up the config
    with open(args['config'], 'r') as f:
        config = json.load(f)
    
    # start time
    config["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
    
    #  create directory structure
    if config['model-uuid'] == "UUID":
        m_uuid = str(uuid.uuid4())[:8]
        config['model-uuid'] = m_uuid
    else:
        m_uuid = config['model-uuid']
        
    folded = 'F' if config['folded'] else ''
    if config['base_directory'] != '':
        base_directory = config['base_directory'] +'/' if config['base_directory'][-1] != '/' else config['base_directory']
        dir_name = base_directory+config['model_type']+'-tf-'+str(config['input_shape'])+'D'+folded+'-'+str(config['var_y'])+'-'+m_uuid
    else:
        dir_name = config['model_type']+'-tf-'+str(config['input_shape'])+'D'+folded+'-'+str(config['var_y'])+'-'+m_uuid
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    config['directory'] = dir_name
    
    # save the config
    with open(config['directory']+'/config-'+config['model-uuid']+'-prelim.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # load data
    df, spark = load_data(config)

    # train the regressor
    logging.info(' running the DNN regressor')
    
    x_test, y_test, regressor, history = runML(df, config)

    # post process the results and save everything
    post_process(regressor, x_test, y_test, history, config)
    
    logging.info(' stopping Spark session')
    spark.stop()
    
    
if __name__ == "__main__":
    
    # execute only if run as a script
    main()

