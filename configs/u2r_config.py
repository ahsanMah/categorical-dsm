import ml_collections

from configs.base_config import get_config as get_base_config
from configs.dataconfigs import get_config as get_data_config

def get_config():
    config = get_base_config()

    # training
    training = config.training
    training.batch_size = 128
    training.n_steps = 1000001

    # data
    config.data = get_data_config("u2r")

    # model
    model = config.model
    model.estimate_noise = True
    model.ndims = 512
    model.time_embedding_size = 128
    model.layers = 16
    model.dropout = 0.0
    model.act = "gelu"

    # optimization
    optim = config.optim
    optim.weight_decay = 1e-5


    return config
