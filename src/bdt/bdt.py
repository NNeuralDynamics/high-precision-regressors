#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import time
from pyspark.sql import SparkSession
from pyspark.sql import dataframe
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import col
from pyspark import SparkConf, SparkContext
import sklearn.model_selection as ms
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
import logging
import json
import uuid
import os
import shutil
import argparse



def eval_training(regressor, config):
    """ Evaluate the training
        argument:
            classifier: the BDT classifier
    """
    results = regressor.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    # plot log loss
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, results['validation_0']['rmse'], label='train')
    plt.plot(x_axis, results['validation_1']['rmse'], label='test')
    plt.yscale('log')
    plt.legend()

    plt.ylabel('rmse')
    plt.title('Regressor rmse')
    plt.grid()

    # plot classification error
    plt.subplot(1, 2, 2)
    plt.plot(x_axis, results['validation_0']['logloss'], label='train')
    plt.plot(x_axis, results['validation_1']['logloss'], label='test')
#     plt.yscale('log')
    plt.legend()

    plt.ylabel('Regression logloss')
    plt.title('Regression logloss')
    plt.grid()
    plt.tight_layout()
    plt.savefig(config['directory']+'/training-evaluation.pdf', dpi=300)
    
    
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
    df = spark.read.options(delimiter=',').schema(schema).format("csv").load(path+str(config['input_shape'])+'D/train/*.csv.*', header='true')

    logging.info(' data loaded into Spark session in {:.3f} seconds'.format(time.time() - start))
    
    # transfer the data to a pandas dataframe
    start = time.time()
    df_p = df.limit(config['sample-size']).toPandas() 

    logging.info(' data loaded into pandas dataframe in {:.3f} seconds'.format(time.time() - start))
    
    return df_p, spark


def runBDT(df, config):
    
    # define variables and target
    var_x = ['x'+str(i+1) for i in range(config['input_shape'])]
    var_y = config['var_y']
    X = df[var_x]
    
    if config['scaled'] == 'normal':
        mu_y = df[var_y].mean()
        sigma_y = df[var_y].std()
        df['y_scaled'] = (df[var_y] - mu_y)/sigma_y
        y = df['y_scaled'].values
        config["mu"] = mu_y
        config["sigma"] = sigma_y
    elif config['scaled'] == 'log':
        df['y_scaled'] = np.log(df[var_y])
        mu_y = df['y_scaled'].mean()
        sigma_y = df['y_scaled'].std()
        df['y_scaled'] = (df['y_scaled'] - mu_y)/sigma_y
        config["mu"] = mu_y
        config["sigma"] = sigma_y
        y = df['y_scaled'].values
    else:
        y = df[var_y].values
        config["mu"] = 0
        config["sigma"] = 1
    
    # Split for training and testing
    x_train, x_test, y_train, y_test = ms.train_test_split(X.values, y, test_size=config["test_size"], random_state=config["random_state"])
    eval_set = [(x_train, y_train), (x_test, y_test)]

    # build and run the regressor
    regressor = xgb.XGBRegressor(max_depth=config["max_depth"], 
                                 learning_rate=config["learning_rate"], 
                                 n_jobs=config["n_jobs"], 
                                 subsample=config["subsample"], 
                                 colsample_bytree=config["colsample_bytree"], 
                                 n_estimators=config["n_estimators"], 
                                 random_state=config["random_state"], 
                                 gamma=config["gamma"],
                                 tree_method=config["tree_method"],
                                 early_stopping_rounds=config["early_stopping_rounds"],
                                 eval_metric=config["eval_metric"])
    
    logging.info(' running the BDT training')
    start = time.time()
    regressor = regressor.fit(x_train, 
                              y_train,
                              eval_set=eval_set,
                              verbose=False)
    config["fit_time"] = timediff(time.time() - start)
    
    return x_test, y_test, regressor


def post_process(regressor, x_test, y_test, config):
    # make pedictions and compute accuracy score
    logging.info(' running the BDT predictions and accuracy computation')
    y_pred = regressor.predict(x_test)
    config["accuracy_score"] = 100*regressor.score(x_test, y_test)
    logging.info(' Accuracy Score: {:4.6f}% '.format(config["accuracy_score"]))
    
    # get the number of estimators
    dump_list = regressor.get_booster().get_dump()
    config["num_trees"] = len(dump_list)

    # evaluate training with plots
    logging.info(' printing training evaluation plots')
    eval_training(regressor, config)
    
    # save model
    logging.info(' saving the trees')
    regressor.save_model(config['directory']+'/bdt-'+str(config['input_shape'])+'D-'+str(config["max_depth"])+'-'+str(config["learning_rate"])+'-'+'{:.1f}'.format(config["sample-size"]/1.e6)+'M.xgb.ubj')
    
    # end time
    config["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())
    
    # save the config
    with open(config['directory']+'/config-'+config['model-uuid']+'.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # remove preliminary config file
    os.remove(config['directory']+'/config-'+config['model-uuid']+'-prelim.json')
    
    # move directory
    shutil.move(config['directory'], config['directory']+'-'+config['scaled']+'-'+str(config['max_depth'])+'-'+str(config['learning_rate'])+f'-{config["accuracy_score"]:.6f}')


def timediff(x):
    return "{}:{}:{}".format(int(x/3600), str(int(x/60%60)).zfill(2), str(round(x - int(x/3600)*3600 - int(x/60%60)*60)).zfill(2))


def main():
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="A python script implementing XGB BDTs for high-precision machine learning",
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
    
    if config['base_directory'] != '':
        base_directory = config['base_directory']+'/' if config['base_directory'][-1] != '/' else config['base_directory']
        dir_name = base_directory+'bdt-xgb-'+str(config['input_shape'])+'D-'+str(config['var_y'])+'-'+m_uuid
    else:
        dir_name = 'bdt-xgb-'+str(config['input_shape'])+'D-'+str(config['var_y'])+'-'+m_uuid
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    config['directory'] = dir_name
    
    # save the config
    with open(config['directory']+'/config-'+config['model-uuid']+'-prelim.json', 'w') as f:
        json.dump(config, f, indent=4)

    # load data
    df, spark = load_data(config)
    
    # run regressor
    x_test, y_test, regressor = runBDT(df, config)

    # post process the results and save everything
    post_process(regressor, x_test, y_test, config)
    
    logging.info(' stopping Spark session')
    spark.stop()

    
if __name__ == "__main__":
    # execute only if run as a script
    main()
