import ml_collections
import math
from configs.base_config import get_config as get_base_config
from configs.dataconfigs import get_config as get_data_config


def get_config():
    config = get_base_config()

    # training
    training = config.training
    training.batch_size = 128
    training.n_steps = 500001
    training.log_freq = 1000
    training.eval_freq = 1000

    # data
    config.data = get_data_config("census")

    # model
    model = config.model
    model.estimate_noise = False
    model.ndims = 512
    model.time_embedding_size = 128
    model.layers = 12
    model.dropout = 0.0
    model.act = "gelu"

    # optimization
    optim = config.optim
    optim.weight_decay = 1e-5

    return config
