import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 256
    training.n_epochs = 100
    training.log_freq = 10
    training.eval_freq = 50
    training.checkpoint_freq = 200
    training.snapshot_freq = 10000

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 1024

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "adult"
    data.categories = [9, 16, 7, 15, 6, 5, 2, 42]
    data.cont_dims = 6

    # model
    config.model = model = ml_collections.ConfigDict()
    model.tau_min = 2.0
    model.tau_max = 10
    model.num_scales = 10
    model.estimate_noise = False
    model.ndims = 128
    model.time_embedding_size = 32
    model.layers = 4
    model.dropout = 0.1
    model.act = "gelu"

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 1e-4
    optim.optimizer = "AdamW"
    optim.lr = 2e-4
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
