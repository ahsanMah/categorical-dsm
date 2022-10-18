import pdb
import pytorch_lightning as pl
from dataloader import get_dataset
from models.resnext import ResNextpp
from models.score_base import ScoreModel


def train(config, workdir):

    model = ScoreModel(config, ResNextpp(config))
    train_loader, val_loader = get_dataset(config)
    trainer = pl.Trainer(
        max_steps=config.training.n_iters,
        gradient_clip_val=config.optim.grad_clip,
        val_check_interval=config.training.eval_freq,
        log_every_n_steps=config.training.log_freq,
    )

    trainer.fit(model, train_loader, val_loader)
