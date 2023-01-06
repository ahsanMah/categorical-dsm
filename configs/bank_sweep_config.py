import ml_collections
import torch
import math
from configs.base_config import get_config as get_base_config
from configs.dataconfigs import get_config as get_data_config

def get_config():
    config = get_base_config()
    # training
    training = config.training
    training.n_steps = 300000
    training.log_freq = 1000
    training.eval_freq = 1000
    training.checkpoint_freq = 1000000
    training.snapshot_freq = 1000000

    # data
    config.data = get_data_config("bank")

    # model
    model = config.model
    model.name = "tab-resnet"

    # Configuration for Hyperparam sweeps
    config.sweep = sweep = ml_collections.ConfigDict()
    param_dict = dict(

        # Regularization components
        optim_weight_decay={
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        },
        model_dropout={"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},

        # Model components
        model_estimate_noise={"values": [True, False]},
        model_embedding_type={"values": ["fourier", "positional"]},
        model_layers={"values": [4, 8, 12, 16]},
        model_ndims={"values": [128, 256, 512, 1024]},
        
        model_act={
            "values": [
                "gelu",
                "relu",
                "selu",
            ]
        },
    )

    sweep.parameters = param_dict
    sweep.method = "bayes"
    sweep.metric = dict(name="val_err")
    sweep.early_terminate = dict(type="hyperband", min_iter=20000, eta=2, s=3)


    return config
