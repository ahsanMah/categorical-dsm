import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 32
    training.n_steps = 1000001
    training.log_freq = 200
    training.eval_freq = 1000
    training.checkpoint_freq = 2000
    training.snapshot_freq = 10000

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 512

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "VOC"
    data.image_size = 32
    data.categorical_channels = 21
    data.continuous_channels = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = "ncsn++"
    model.tau_min = 2.0
    model.tau_max = 20.0
    model.num_scales = 20
    model.estimate_noise = False
    model.act = 'swish'
    model.embedding_type = 'fourier'

    # Modified from CIFAR-10 NCSN config
    model.scale_by_sigma = False
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 8
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3
    model.dropout = 0.

    config.msma = msma = ml_collections.ConfigDict()
    msma.denoise = False

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "AdamW"
    optim.lr = 2e-4
    optim.grad_clip = 1.0
    optim.scheduler = "none"

    config.devtest=False
    config.seed = 42
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
    elif torch.backends.mps.is_built():
        config.device = torch.device("mps")
    else:
        config.device = torch.device("cpu")

    return config