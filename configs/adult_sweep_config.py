import ml_collections
import torch
import math
from configs.adult_config import get_config as get_base_config


def get_config():
    config = get_base_config()
    # training
    training = config.training
    training.n_steps = 200000
    training.log_freq = 100
    training.eval_freq = 100
    training.checkpoint_freq = 1000000
    training.snapshot_freq = 1000000

    # model
    model = config.model
    model.name = "tab-resnet"

    # Configuration for Hyperparam sweeps
    config.sweep = sweep = ml_collections.ConfigDict()
    param_dict = dict(
        # optim_optimizer={"values": ["Adam", "Adamax", "AdamW"]},
        # optim_lr={
        #     "distribution": "log_uniform",
        #     "min": math.log(1e-5),
        #     "max": math.log(1e-2),
        # },
        # Regularization components
        optim_weight_decay={
            "distribution": "log_uniform",
            "min": math.log(1e-6),
            "max": math.log(1e-2),
        },
        model_dropout={"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        model_ema_rate={"values": [0.999, 0.9999]},
        # Model components
        model_time_embedding_sz={"values": [64, 128, 256]},
        model_embedding_type={"values": ["fourier", "positional"]},
        model_layers={"values": [2, 4, 6, 8, 12]},
        # model_ndims={"values": [256, 512, 1024]},
        model_act={
            "values": [
                "gelu",
                "relu",
                "swish",
                "selu",
                "softplus",
                "mish",
                "prelu",
                "elu",
            ]
        },
    )

    sweep.parameters = param_dict
    sweep.method = "random"
    # sweep.metric = dict(name="val_loss")

    return config
