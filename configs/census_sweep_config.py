import ml_collections
import torch
import math
from configs.base_config import get_config as get_base_config
from configs.dataconfigs import get_config as get_data_config


def get_config():
    config = get_base_config()
    # training
    training = config.training
    training.batch_size = 2048
    training.n_steps = 150001
    training.log_freq = 200
    training.eval_freq = 200
    training.checkpoint_freq = 10000000
    training.snapshot_freq = 10000000

    # data
    config.data = get_data_config("census")

    # model
    model = config.model
    model.name = "tab-resnet"
    model.ndims = 1024
    model.layers = 20

    # Configuration for Hyperparam sweeps
    config.sweep = sweep = ml_collections.ConfigDict()
    param_dict = dict(
        # Regularization components
        optim_weight_decay={
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        },
        
        # Taken from original Adam paper
        # Section 6.4 https://arxiv.org/pdf/1412.6980.pdf
        optim_lr={"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
        optim_beta1={"distribution": "uniform", "min":0.0, "max": 0.9},
        # optim_beta2={"values": [0.99, 0.999, 0.9999]},

    )

    sweep.parameters = param_dict
    sweep.method = "random"
    # sweep.metric = dict(name="val_err")
    # # Brackets would be [20k, 20k*eta, 20k*eta**2, 20k*eta**3]
    # sweep.early_terminate = dict(type="hyperband", min_iter=20000, eta=2, s=3)

    return config
