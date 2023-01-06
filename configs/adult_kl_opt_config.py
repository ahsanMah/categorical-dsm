import ml_collections
import torch
import math

def get_config():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 128
    training.n_steps = 500000
    # training.n_epochs = 1000
    training.log_freq = 100
    training.eval_freq = 100
    training.checkpoint_freq = 2000
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
    model.name = "tab-resnet"
    model.tau_min = 2.0
    model.tau_max = 20
    model.sigma_min = 1e-1
    model.sigma_max = 1.0
    model.num_scales = 10
    model.estimate_noise = True
    model.ndims = 512
    model.time_embedding_size = 128
    model.layers = 8
    model.dropout = 0.3
    model.act = "swish"
    model.embedding_type = "fourier"
    model.ema_rate = 0.999

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 3e-5
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

    # Configuration for Hyperparam sweeps
    config.sweep = sweep = ml_collections.ConfigDict()
    param_dict = dict(

        # Regularization components
        optim_weight_decay={
            "distribution": "log_uniform",
            "min": math.log(1e-6),
            "max": math.log(1e-2),
        },
        model_dropout={"values": [0.0, 0.1, 0.2, 0.3]},
        # Model components
        model_time_embedding_sz={"values": [64, 128, 256]},
        model_embedding_type={"values": ["fourier", "positional"]},
        model_layers={"values": [4, 6, 8, 12]},
        model_act={
            "values": [
                "gelu",
                "swish",
            ]
        },
    )

    sweep.parameters = param_dict
    sweep.method = "bayes"
    sweep.metric = dict(name="val_loss")
    sweep.early_terminate = dict(type="hyperband", min_iter=50000, eta=2, s=3)

    return config
