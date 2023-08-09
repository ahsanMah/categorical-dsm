import ml_collections

from configs.base_config import get_config as get_base_config
from configs.dataconfigs import get_config as get_data_config

def get_config():
    config = get_base_config()

    # training
    training = config.training
    training.batch_size = 512
    training.n_steps = 500001

    # data
    config.data = get_data_config("cars")

    # model
    model = config.model
    model.name = "tab-transformer"
    model.estimate_noise = True
    model.time_embedding_size = 128
    model.ndims = 256
    model.layers = 8
    model.dropout = 0.0
    model.attention_heads = 8
    model.attention_dim_head = 128

    # optimization
    optim = config.optim
    optim.weight_decay = 0.0
    optim.lr = 3e-4
    optim.scheduler = "none"


    return config
