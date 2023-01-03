import os

import numpy as np
import pytorch_lightning as pl
import torch

from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torchinfo import summary

import wandb
from dataloader import get_dataset
from models.score_base import VisionScoreModel, TabScoreModel
import logging


def train(config, workdir):

    pl.utilities.seed.seed_everything(config.seed)

    if "tab" in config.model.name:
        model = TabScoreModel(config)
    else:
        model = VisionScoreModel(config)

    train_loader, val_loader, test_loader = get_dataset(config)

    # Checkpoint that saves periodically to allow for resuming later
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{workdir}/checkpoints-meta/",
        save_last=True,  # Saves a copy as `last.ckpt`
        every_n_train_steps=config.training.checkpoint_freq,
    )

    snapshot_callback = ModelCheckpoint(
        dirpath=f"{workdir}/checkpoints/",
        monitor="val_loss",
        filename="{step}-{val_loss:.2f}",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=config.training.snapshot_freq,
    )

    callback_list = [checkpoint_callback, snapshot_callback]

    if "tab" in config.model.name:
        logging.info(ModelSummary(model, max_depth=2))
    else:
        summary(
            model,
            depth=3,
            input_data=[
                torch.empty(
                    1,
                    config.data.num_categories,
                    config.data.image_size,
                    config.data.image_size,
                ),
                torch.zeros(
                    1,
                ),
            ],
        )

    wandb.watch(model, log_freq=config.training.snapshot_freq, log="all")
    wandb_logger = WandbLogger(log_model=False, save_dir="wandb")
    tb_logger = TensorBoardLogger(
        save_dir=f"{workdir}/tensorboard_logs/", name="", default_hp_metric=False
    )

    trainer = pl.Trainer(
        accelerator=str(config.device),
        default_root_dir=workdir,
        # max_epochs=config.training.n_epochs,
        max_steps=config.training.n_steps,
        gradient_clip_val=config.optim.grad_clip,
        val_check_interval=config.training.eval_freq,
        log_every_n_steps=config.training.log_freq,
        callbacks=callback_list,
        fast_dev_run=5 if config.devtest else 0,
        enable_model_summary=False,
        logger=[tb_logger, wandb_logger],
        # num_sanity_val_steps=0,
    )
    # ckpt_path = f"{workdir}/checkpoints-meta/last.ckpt"
    # if not os.path.exists(ckpt_path):
    ckpt_path = None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


def eval(config, workdir, ckpt_num=-1, denoise=False):

    # import torch.nn.functional as F
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import numpy as np
    # import pandas as pd
    # from torchinfo import summary
    # import seaborn as sns
    # import matplotlib as mpl

    # mpl.rc("figure", figsize=(10, 4), dpi=100)
    # sns.set_theme()

    ckpt_dir = os.path.join(workdir, "checkpoints")
    ckpts = sorted(os.listdir(ckpt_dir))
    ckpt = ckpts[ckpt_num]
    scorenet = TabScoreModel.load_from_checkpoint(
        checkpoint_path=os.path.join(ckpt_dir, ckpt), config=config
    ).cuda()
    scorenet.eval()

    train_loader, val_loader, test_loader = get_dataset(config, train_mode=False)
    outdict = {}
    with torch.cuda.device(0):
        for ds, loader in [("val", val_loader), ("test", test_loader)]:
            score_norms = []
            labels = []
            for x_batch, y in loader:
                s = scorenet.scorer(x_batch.cuda(), denoise_step=denoise).cpu().numpy()
                score_norms.append(s)
                labels.append(y.numpy())
            score_norms = np.concatenate(score_norms)
            labels = np.concatenate(labels)
            outdict[ds] = {"score_norms": score_norms, "labels": labels}

    step = ckpt.split("-")[0]
    os.makedirs(os.path.join(workdir, "score_norms"), exist_ok=True)
    fname = os.path.join(
        workdir, "score_norms", f"{step}-{'denoise' if denoise else ''}-score_norms.npz"
    )

    with open(fname, "wb") as f:
        np.savez_compressed(f, **outdict)

    # from ood_detection_helper import ood_metrics, auxiliary_model_analysis

    # X_train = outdict["val"]["score_norms"]
    # test_labels = outdict["test"]["labels"]
    # X_test = outdict["test"]["score_norms"][test_labels == 0]
    # X_ano = outdict["test"]["score_norms"][test_labels == 1]
    # results = auxiliary_model_analysis(X_train, X_test, [X_ano],
    #                                 components_range=range(5,6),
    #                                 labels=["Train", "Inlier", "Outlier"])
