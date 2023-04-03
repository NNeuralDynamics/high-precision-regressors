#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import dataframe
from pyspark.sql.types import StructType, StructField, DoubleType
from sklearn import metrics
import time
import sys
import logging
import uuid
import os
import shutil
import json
import argparse
import pytorch_model_summary as pms

SEED = 42
CLIP = 1e2

#############################
# Data Stuff
#############################

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
    
    logger = spark._jvm.org.apache.log4j
    logging.getLogger("py4j.clientserver").setLevel(logging.WARN)
    
    return df_p, spark


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
    

class df_to_tensor(torch.utils.data.Dataset):
 
    def __init__(self, df, config):
        self.df = df.copy(deep = True)

        # define variables and target
        var_x = ['x'+str(i+1) for i in range(config['input_shape'])]
        var_y = config['var_y']
        x = df[var_x].values

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


        self.x = torch.tensor(x, dtype=torch.float64).to(get_device())
        self.y = torch.tensor(y, dtype=torch.float64).reshape(-1, 1).to(get_device())
 
    def __len__(self):
        return len(self.x)
   
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]


def build_data_loaders(df, config):
    # slice and dice the data into train, validation and test sets
    df_test_data =  df.sample(frac=config["test_split"], random_state=SEED)
    df_validation_data = df.drop(df_test_data.index).sample(frac=config["validation_split"], random_state=SEED)
    df_train_data = df.drop(df_test_data.index).drop(df_validation_data.index).sample(frac=1, random_state=SEED)

    test_data = df_to_tensor(df_test_data, config)
    validation_data = df_to_tensor(df_validation_data, config)
    train_data = df_to_tensor(df_train_data, config)

    # load the data into torch tensor batches
    batch = config["batch_size"]
    numworkers = 2 if get_device().type == 'cpu' else 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=numworkers)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch, shuffle=True, num_workers=numworkers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=False, num_workers=numworkers)
    
    return train_loader, validation_loader, test_loader

#############################
# ML Stuff
#############################

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")
        
    return device

def init_torch():
    
    gen = torch.manual_seed(SEED)
    
    device = get_device()

    if device.type == 'cpu':
        torch.set_num_threads(24)
        torch.set_num_interop_threads(1)

    logging.info(f' torch is using the {device}')
    
    return device

class EarlyStopping:
    def __init__(self, m_path, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.m_path = m_path
        
    def reset_counter(self):
        self.counter = 0

    def early_stop(self, model, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            torch.save(model.state_dict(), self.m_path)
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    
class skip_block(torch.nn.Module):
    """ the basic building block of a dnn with skip connections
        arguments:
            x: the input
            width: the width of the hidden layers
            activation: the activation function: 'relu', 'elu', 'swish' (silu), 'leaky_relu', 'softplus'
            squeeze: a boolean specifying wheher the skip units are squeezed
        returns:
            res: the skip net block
    """
    def __init__(self, input_shape, width, activation, n_layers=1):
        super(skip_block, self).__init__()
        self.input_shape = input_shape
        self.width = width
        
        self.input = torch.nn.Linear(self.input_shape, self.width)
        self.fc_module = torch.nn.ModuleList([torch.nn.Linear(self.width, self.width) for i in range(n_layers)])
        self.linear = torch.nn.Linear(self.width, self.width)
        if self.input_shape != self.width:
            self.reshape = torch.nn.Linear(self.input_shape, self.width, bias=False)
        
        self.act = activation
        
    def forward(self, x):
        y = self.act(self.input(x))
        for l in self.fc_module:
            y = self.act(l(y))
        y = self.linear(y)
        if self.input_shape != self.width:
            residual = self.reshape(x)
        else:
            residual = x
        y += residual
        
        return self.act(y)
    
    
class skip_dnn(torch.nn.Module):
    def __init__(self, sk_block, config):
        super(skip_dnn, self).__init__()
        self.width = config["width"]
        self.n_blocks = config["depth"] - 1
        
        if config["activation"] == 'leaky_relu':
            self.act = torch.nn.LeakyReLU()
        elif config["activation"] == 'relu':
            self.act = torch.nn.ReLU()
        if config["activation"] == 'softplus':
            self.act = torch.nn.Softplus()
        if config["activation"] == 'swish':
            self.act = torch.nn.SiLU()
        
        self.input = skip_block(config["input_shape"], self.width, self.act)
        self.core = self.make_layers(skip_block)
        self.output = torch.nn.Linear(self.width, 1)
            
    def make_layers(self, skip_block):
        layers = []
        for bl in range(self.n_blocks):
            layers.append(skip_block(self.width, self.width, self.act))
        return torch.nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.input(x)
        x = self.core(x)
        return self.output(x)
    
    
class dnn(torch.nn.Module):
    def __init__(self, config):
        super(dnn, self).__init__()
        self.width = config["width"]
        
        self.input = torch.nn.Linear(config["input_shape"], self.width)
        self.fc_module = torch.nn.ModuleList([torch.nn.Linear(self.width, self.width) for i in range(config["depth"] - 1)])
        self.output = torch.nn.Linear(self.width, 1)
        
        if config["activation"] == 'leaky_relu':
            self.act = torch.nn.LeakyReLU()
        elif config["activation"] == 'relu':
            self.act = torch.nn.ReLU()
        if config["activation"] == 'softplus':
            self.act = torch.nn.Softplus()
        if config["activation"] == 'swish':
            self.act = torch.nn.SiLU()
        
    def forward(self, x):
        x = self.act(self.input(x))
        for l in self.fc_module:
            x = self.act(l(x))
        return self.output(x)

    
def nets(config):
    """ the tensorflow model builder
        arguments:
            config: the configuration file
        returns:
            regressor: the tensorflow model
    """
    
    # define the tensorflow model
    if config["model_type"] == 'dnn':
        regressor = dnn(config).double().to(get_device())
    elif config["model_type"] == 'skip':
        regressor = skip_dnn(skip_block, config).double().to(get_device())
    else:
        logging.error(' '+config["model_type"]+' not implemented. model_type can be either dnn, skip or squeeze')
        
        
    # save parameter counts
    summary = pms.summary(regressor, torch.tensor(torch.zeros((config["input_shape"],)).to(get_device())).double()).rstrip().split('\n')
    config["trainable_parameters"] = int(summary[-3].replace(',', '')[18:])
    config["non_trainable_parameters"] = int(summary[-2].replace(',', '')[22:])
    config["total_parameters"] = int(summary[-4].replace(',', '')[14:])
    
    # save config
    with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    return regressor


def train_one_epoch(model, train_data, f_optimizer, f_loss, max_steps):
    epoch_loss = 0.

    for i, data in enumerate(train_data):
        
        # steps per epoch
        if i == max_steps:
            break
        
        # forward prop
        x, y = data
        f_optimizer.zero_grad()
        y_out = model(x)

        # backprop
        loss = f_loss(y_out, y)
        if loss.item() > CLIP:
            return loss.item()
        loss.backward()
        f_optimizer.step()

        # Gather data and report
        epoch_loss += loss.item()

    return epoch_loss/(i + 1)

def validate_one_epoch(model, validate_data, f_loss, max_steps):
    epoch_loss = 0.
    abs_score = 0.
    r2_score = 0.
    
    for i, data in enumerate(validate_data):
        if i == max_steps:
            break
        
        # forward prop
        x, y = data
        y_pred = model(x)
        
        # loss
        loss = f_loss(y_pred, y)
        epoch_loss += loss.item()
        
        # accuracy
        abs_score += (1 - np.mean(np.abs( ((y_pred - y)/y).cpu().detach().numpy() )))*100
        r2_score += metrics.r2_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())*100

    return epoch_loss / (i + 1), abs_score / (i + 1), r2_score / (i + 1)


def test_model(model, test_data):
    abs_score = 0.
    r2_score= 0.
    
    for i, data in enumerate(test_data):
        
        # forward prop
        x, y = data
        y_pred = model(x)
        
        # accuracy
        abs_score += (1 - np.mean(np.abs( ((y_pred - y)/y).cpu().detach().numpy() )))*100
        r2_score += metrics.r2_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())*100


    return abs_score / (i + 1), r2_score / (i + 1)

    
def runML(df, config):
    """ run the DNN
        arguments:
            df: the dataframe including, training, validation and test
            config: the configuration dictionary for the hyperparameters
    """

    train_loader, validation_loader, test_loader = build_data_loaders(df,config)
    
    regressor = nets(config)

    # print the summary
    logging.info('\n' + pms.summary(regressor, torch.tensor(torch.zeros((config["input_shape"],)).to(get_device())).double()))
    
    # define the loss function
    if config['loss'] == 'mse':
        loss_fn = torch.nn.MSELoss()
    elif config['loss'] == 'mae':
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError('loss type not defined. Has to be mae or mse')
            
    # define the optimizer
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)
    
    # learning rate decay
    if config['lr_decay_type'] == 'exp':
        gamma = (0.5 ** (10000000 / config["decay_steps"])) ** (1 / 2500)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=optimizer,
                        gamma=gamma
                    )
    elif config['lr_decay_type'] == 'poly':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
                        optimizer=optimizer,
                        total_iters=config['decay_steps'],
                        power=0.5
                    )
    elif config['lr_decay_type'] == 'const':
        scheduler = torch.optim.lr_scheduler.ConstantLR(
                        optimizer=optimizer,
                        factor=1., 
                        total_iters=config["epochs"]
                    )
    else:
        raise ValueError('lr type not defined. Has to be exp or poly or const')
        
    # the training history
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": [], "abs_score": [], "r2_score": []}
    
    # model storage
    model_path = config['directory'] + f'/dnn-{config["depth"]}-{config["width"]}-{config["activation"]}-adam-{config["lr_decay_type"]}-schedule-{config["loss"]}-{config["monitor"]}.torch'
    
    # early stopping
    early_stopping = EarlyStopping(model_path, patience=config["patience"])
    
    
    # run the regressor
    start = time.time()
    epoch = 0
    while epoch < config["epochs"]:

        # Make sure gradient tracking is on, and do a pass over the data
        regressor.train(True)
        avg_loss = train_one_epoch(regressor, train_loader, optimizer, loss_fn, config["steps_per_epoch"])
        
        # for some activations there are spikes in the loss function: skip the epoch when that happens
        if avg_loss > CLIP:
            regressor.train(False)
            early_stopping.reset_counter()
            regressor.load_state_dict(torch.load(model_path))
            continue
            
        # add to history
        history["epoch"].append(epoch + 1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(avg_loss)

        # We don't need gradients on to do reporting
        regressor.train(False)
        avg_vloss, abs_score, r2_score = validate_one_epoch(regressor, validation_loader, loss_fn, config["steps_per_epoch"])
        history["val_loss"].append(avg_vloss)
        history["abs_score"].append(abs_score)
        history["r2_score"].append(r2_score)
        
        logging.info(f' Epoch {epoch + 1}: training loss = {avg_loss:.8f}  validation loss = {avg_vloss:.8f}  learning rate = {optimizer.param_groups[0]["lr"]:0.6e}  relative accuracy: {abs_score:.6f}  R2 score: {r2_score:.6f}')
        
        # check for early stopping
        if early_stopping.early_stop(regressor, avg_vloss):
            regressor.load_state_dict(torch.load(model_path))
            break

        # if early stopping did not happen revert to the best model
        if epoch == config["epochs"] - 1:
            regressor.load_state_dict(torch.load(model_path))
            
        # decay learning rate
        scheduler.step()
        
        epoch += 1
        
    config["fit_time"] = timediff(time.time() - start)
    
    return test_loader, regressor, history


#############################
# utils
#############################

def timediff(x):
    """ a function to convert seconds to hh:mm:ss
        argument:
            x: time in seconds
        returns:
            time in hh:mm:ss
    """
    return "{}:{}:{}".format(int(x/3600), str(int(x/60%60)).zfill(2), str(round(x - int(x/3600)*3600 - int(x/60%60)*60)).zfill(2))


def plot_loss(history, dir_name):
    """ plotting routine
        argument:
            history: the tf history object
    """
    plt.plot(history['train_loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('y')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/training-evaluation.pdf', dpi=300)


def post_process(regressor, test_loader, history, config):
    """ post process the regressor to check for accuracy and save everything
        argumants:
            regressor: the tensorflow regressor object
            history: the history object for the training
            config: the configuration for the training
    """
    # check accuracy
    logging.info(' running the DNN predictions and accuracy computation')
    
    abs_score, r2_score = test_model(regressor, test_loader)
    config["abs_score"] = abs_score
    config["r2_score"] = r2_score
    logging.info(' relative accuracy: {:.6f}%  |---|  R2 score: {:.6f}%'.format(abs_score, r2_score))
        
    #plot the training history
    logging.info(' printing training evaluation plots')
    plot_loss(history, config['directory'])
    
    # end time
    config["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
                 
    # save config
    with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # save history
    with open(config['directory']+'/history-'+config['model-uuid']+'.json', 'w') as f:
        json.dump(history, f, indent=4)
        
    # remove preliminary config file
    os.remove(config['directory']+'/config-'+config['model-uuid']+'-prelim.json')
        
    # move directory
    shutil.move(config['directory'], config['directory']+'-'+config['scaled']+'-'+str(config['depth'])+'-'+str(config['width'])+'-'+config['activation']+'-'+str(config['batch_size'])+'-adam-'+config['lr_decay_type']+'-schedule-'+config['loss']+'-'+config['monitor']+f'-{abs_score:.6f}-{r2_score:.6f}')    
    
    
#############################
# main
#############################
    
def main():
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = init_torch()
    
    parser = argparse.ArgumentParser(description="A python script implementing DNN and sk-DNN for high-precision machine learning",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", help="configuration file for the run")
    args = vars(parser.parse_args())
    
    # set up the config
    with open(args['config'], 'r') as f:
        config = json.load(f)
    
    # start time
    config["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
    
    # set device
    config["device"] = device.type + ':' + device.index if device.index else device.type
    
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
    
    test_loader, regressor, history = runML(df, config)

    # post process the results and save everything
    post_process(regressor, test_loader, history, config)
    
    logging.info(' stopping Spark session')
    spark.stop()
    
    
if __name__ == "__main__":
    
    # execute only if run as a script
    main()

