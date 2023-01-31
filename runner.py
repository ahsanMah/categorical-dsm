import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from torchinfo import summary

import wandb
from dataloader import get_dataset
from models.score_base import TabScoreModel, VisionScoreModel
from ood_detection_helper import auxiliary_model_analysis, ood_metrics

mpl.rc("figure", figsize=(10, 4), dpi=100)
sns.set_theme()


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
        filename="{step}-{val_loss:.4f}",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=config.training.snapshot_freq,
    )

    callback_list = [checkpoint_callback, snapshot_callback]

    if "tab" in config.model.name:
        logging.info(ModelSummary(model, max_depth=3))
    else:
        summary(
            model,
            depth=3,
            input_data=[
                torch.zeros(
                    1,
                    config.data.categorical_channels + config.data.continuous_channels,
                    config.data.image_size,
                    config.data.image_size,
                ),
                torch.zeros(
                    1,
                ),
            ],
        )

    # wandb.watch(model, log_freq=config.training.snapshot_freq, log="all")
    wandb_logger = WandbLogger(log_model=False, save_dir="wandb")
    tb_logger = TensorBoardLogger(
        save_dir=f"{workdir}/tensorboard_logs/", name="", default_hp_metric=False
    )

    ckpt_path = f"{workdir}/checkpoints-meta/last.ckpt"
    if not os.path.exists(ckpt_path):
        ckpt_path = None

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
        check_val_every_n_epoch=None,
        logger=[tb_logger, wandb_logger],
        # num_sanity_val_steps=0,
        resume_from_checkpoint=ckpt_path,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # eval(config, workdir, ckpt_num=-1)


def eval(config, workdir, ckpt_num=-1):

    denoise = config.msma.denoise
    ckpt_dir = os.path.join(workdir, "checkpoints-meta")
    ckpts = sorted(os.listdir(ckpt_dir))
    ckpt = ckpts[ckpt_num]
    step = ckpt.split("-")[0]
    fname = os.path.join(
        workdir, "score_norms", f"{step}-{'denoise' if denoise else ''}-score_norms.npz"
    )

    print(
        f"Evaluating {ckpt} with denoise = {denoise} and saving to {fname} if not already present."
    )

    if os.path.exists(fname):
        print(f"Loading from {fname}")
        with np.load(fname, allow_pickle=True) as npzfile:
            outdict = {k: npzfile[k].item() for k in npzfile.files}
    else:
        scorenet = TabScoreModel.load_from_checkpoint(
            checkpoint_path=os.path.join(ckpt_dir, ckpt), config=config
        ).cuda()
        scorenet.eval()

        train_loader, val_loader, test_loader = get_dataset(config, train_mode=False)
        outdict = {}
        with torch.cuda.device(0):
            for ds, loader in [
                ("train", train_loader),
                ("val", val_loader),
                ("test", test_loader),
            ]:
                score_norms = []
                labels = []
                for x_batch, y in loader:
                    s = (
                        scorenet.scorer(x_batch.cuda(), denoise_step=denoise)
                        .cpu()
                        .numpy()
                    )
                    score_norms.append(s)
                    labels.append(y.numpy())
                score_norms = np.concatenate(score_norms)
                labels = np.concatenate(labels)
                outdict[ds] = {"score_norms": score_norms, "labels": labels}

        os.makedirs(os.path.join(workdir, "score_norms"), exist_ok=True)
        fname = os.path.join(
            workdir,
            "score_norms",
            f"{step}-{'denoise' if denoise else ''}-score_norms.npz",
        )

        with open(fname, "wb") as f:
            np.savez_compressed(f, **outdict)

    X_train = outdict["train"]["score_norms"]
    np.random.seed(42)
    np.random.shuffle(X_train)
    X_val = outdict["val"]["score_norms"]
    X_train = np.concatenate((X_train[: len(X_val)], X_val))
    test_labels = outdict["test"]["labels"]
    X_test = outdict["test"]["score_norms"][test_labels == 0]
    X_ano = outdict["test"]["score_norms"][test_labels == 1]
    results = auxiliary_model_analysis(
        X_train,
        X_test,
        [X_ano],
        components_range=range(5, 6, 1),
        labels=["Train", "Inlier", "Outlier"],
    )
    ood_metrics(
        -results["GMM"]["test_scores"],
        -results["GMM"]["ood_scores"][0],
        plot=True,
        verbose=True,
    )
    plt.suptitle(f"{config.data.dataset} - GMM", fontsize=18)
    plt.savefig(fname.replace("score_norms.npz", "gmm.png"), dpi=100)

    ood_metrics(
        results["KD"]["test_scores"],
        results["KD"]["ood_scores"][0],
        plot=True,
        verbose=True,
    )
    plt.suptitle(f"{config.data.dataset} - KD Tree", fontsize=18)
    plt.savefig(fname.replace("score_norms.npz", "kd.png"), dpi=100)

    logging.info(results["GMM"]["metrics"])
