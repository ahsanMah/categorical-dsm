import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 256
    training.n_steps = 1000000
    training.log_freq = 200
    training.eval_freq = 200
    training.checkpoint_freq = 2000
    training.snapshot_freq = 10000

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 1024

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "MNIST"
    data.image_size = 28
    data.num_categories = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = "resnext"
    model.tau_min = 2.0
    model.tau_max = 20
    model.num_scales = 10
    model.estimate_noise = False
    model.nf = 128
    model.time_embedding_size = 64
    model.layers = (2, 4, 4, 4, 2)
    model.dropout = 0.0
    model.act = "gelu"
    model.ema_rate = 0.9999
    model.embedding_type = "fourier"

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0
    optim.optimizer = "AdamW"
    optim.lr = 3e-4
    optim.grad_clip = 1.0
    optim.scheduler = None

    config.devtest = False
    config.seed = 42
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
    elif torch.backends.mps.is_built():
        config.device = torch.device("mps")
    else:
        config.device = torch.device("cpu")

    return config
