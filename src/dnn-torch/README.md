The configuration file needs to be specified as such:

```json
{
    "model_type": "dnn",
    "input_shape": 8,
    "var_y": "yRT",
    "depth": 8,
    "width": 100,
    "data_path": "../../data/8D/",
    "scaled": "normal",
    "lr_decay_type": "exp",
    "initial_lr": 0.001,
    "final_lr": 1e-06,
    "decay_steps": 1000000,
    "validation_split": 0.4,
    "test_split": 0.2,
    "batch_size": 512,
    "steps_per_epoch": 4000,
    "patience": 200,
    "monitor": "val_mse",
    "loss": "mse",
    "metrics": [
        "mse",
        "mae"
    ],
    "verbose": 0,
    "base_directory": "",
    "epochs": 10000,
    "sample-size": 10000000,
    "model-uuid": "UUID"
}
```

__model_type__: _dnn_ or _skip_ which refer to normal DNN or DNN with skip connections  
__input_shape__: _2_, _4_ or _8_ for 2D, 4D or 8D respectively  
__var_y__: the name of the target variable. This can be set to "yN" for normalizer or "y_2" for non-normalized  
__depth__: the depth of the neural network. For network with skip connections it is the number of blocks  
__width__: the width of the network  
__data_path__: the parth to the relevant data (please check the data folder for instructions)  
__scaled__: _normal_, _log_ or _none_. For details please refer to the paper  
__lr_decay_type__: _exp_, _poly_, _const_ for exponentially decaying, polynomially decaying and constant learning rate  
__initial_lr__: the initial learning rate  
__final_lr__: the final laerning rate (for exponentially decaying learning rate)  
__decay_steps__: the stepsize for the learning rate decay  
__validation_split__: the split into train and validation data, both used in training  
__validation_split__: the split for the test set (fraction of total data)  
__batch_size__: the batch size for the stochastic gradient descent   
__steps_per_epoch__: the steps per epoch (as defined in Tensorflow)  
__patience__: the number of epochs with no improvement in the metrics after which the training is stopped  
__monitor__: the metric to be monitored for early stopping  
__loss__: the loss function  
__metrics__: the metrics to be reported and monitored during training  
__verbose__: _0_ or _1_ for the verbosity level  
__base_dir__: any necessary prefixes for the folder in which the results and models will be stored  
__epochs__: the total number of epochs in the training phase  
__sample-size__: the total sample size including train ((1-validation_split) x 80%), validation (validation_split x 80%) and test (20%) samples  
__model-uuid__: a uuid generated for each run which can be specified of left to "UUID" to be generated during the run  


###  to run the code:
```sh
python3 dnn.py config.json
```
