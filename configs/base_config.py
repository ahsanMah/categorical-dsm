import ml_collections
import torch
import math

def get_config():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 128
    training.n_steps = 500001
    training.log_freq = 1000
    training.eval_freq = 1000
    training.checkpoint_freq = 2000
    training.snapshot_freq = 10000
    training.resume=False

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 1024

    # data config will be added later
    config.data = data = ml_collections.ConfigDict()

    # default model parameters
    config.model = model = ml_collections.ConfigDict()
    model.name = "tab-resnet"
    model.tau_min = 2.0
    model.tau_max = 20
    model.sigma_min = 1e-1
    model.sigma_max = 1.0
    model.num_scales = 20
    model.estimate_noise = True
    model.ndims = 512
    model.time_embedding_size = 128
    model.layers = 12
    model.dropout = 0.0
    model.act = "gelu"
    model.embedding_type = "fourier"
    model.ema_rate = 0.999

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0
    optim.optimizer = "AdamW"
    optim.lr = 3e-4
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.grad_clip = 1.0
    optim.scheduler = "none"

    config.msma = msma = ml_collections.ConfigDict()
    msma.denoise = True
    msma.checkpoint = "best"

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
        model_dropout={"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        # Model components
        model_ndims={"values": [256, 512, 1024]},
        # model_time_embedding_sz={"values": [64, 128, 256]},
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
