import pdb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import get_dataset
from models.resnext import ResNextpp
from models.score_base import ScoreModel


def train(config, workdir):

    model = ScoreModel(config, ResNextpp(config))
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

    trainer = pl.Trainer(
        accelerator=str(config.device),
        default_root_dir=workdir,
        max_epochs=config.training.n_epochs,
        gradient_clip_val=config.optim.grad_clip,
        val_check_interval=config.training.eval_freq,
        log_every_n_steps=config.training.log_freq,
        callbacks=[checkpoint_callback, snapshot_callback],
    )

    trainer.fit(model, train_loader, val_loader)
