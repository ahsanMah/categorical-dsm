import os

import pytorch_lightning as pl
import torch

from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

import wandb
from dataloader import get_dataset
from models.score_base import ScoreModel, TabScoreModel

def train(config, workdir):

    if "tab" in config.model.name:
        model = TabScoreModel(config)
    else:
        model = ScoreModel(config)
    
    train_loader, val_loader = get_dataset(config)

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

    # summary(
    #     model,
    #     input_data=[
    #         torch.empty(
    #             1,
    #             config.data.num_categories,
    #             config.data.image_size,
    #             config.data.image_size,
    #         ),
    #         torch.zeros(
    #             1,
    #         ),
    #     ],
    # )
    
    print(ModelSummary(model, max_depth=2))

    wandb.watch(model, log_freq=config.training.snapshot_freq//1000, log="all")
    wandb_logger = WandbLogger(log_model=False, save_dir="wandb")

    trainer = pl.Trainer(
        accelerator=str(config.device),
        default_root_dir=workdir,
        max_epochs=config.training.n_epochs,
        gradient_clip_val=config.optim.grad_clip,
        val_check_interval=config.training.eval_freq,
        log_every_n_steps=config.training.log_freq,
        callbacks=[checkpoint_callback, snapshot_callback],
        fast_dev_run=5 if config.devtest else 0,
        enable_model_summary=False,
        logger=wandb_logger,
        # num_sanity_val_steps=0,
    )

    # ckpt_path = f"{workdir}/checkpoints-meta/last.ckpt"
    # if not os.path.exists(ckpt_path):
    ckpt_path = None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
