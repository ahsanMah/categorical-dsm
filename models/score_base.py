from distutils.command.config import config
import pdb
import ml_collections
import pytorch_lightning as pl
import torch
from losses import categorical_dsm_loss

from models.mutils import get_taus, log_concrete_sample, prob_to_logit, get_optimizer


class ScoreModel(pl.LightningModule):
    def __init__(self, config: ml_collections.ConfigDict, net: torch.nn.Module) -> None:
        super().__init__()

        self.model_config = config.model
        for model_attr, val in config.model.to_dict().items():
            setattr(self, model_attr, val)

        self.net = net
        self.training_opts = config.training
        self.optimization_opts = config.optim
        self.K = config.data.num_categories

        self.register_buffer("taus", torch.tensor(get_taus(config), device=self.device))

        # self.save_hyperparameters()

    def configure_optimizers(self):

        optim = get_optimizer(self.optimization_opts.optimizer)
        optimizer = optim(
            self.parameters(),
            lr=self.optimization_opts.lr,
            weight_decay=self.optimization_opts.weight_decay,
        )

        if self.optimization_opts.scheduler is not None:
            scheduler = torch.lr_scheduler.StepLR(
                step_size=int(0.3 * self.training_opts.n_epochs), gamma=0.3
            )
            return optimizer, scheduler

        return optimizer

    def forward(self, x, t):

        if self.estimate_noise:
            """Assumes that the model is predicting the `logit noise`
            Following the log concrete gradient, we need to rescale
            the logits to turn them into scores
            """
            tau = self.taus[t]
            logit_noise_est = self.net(x, t)
            score = (
                tau * self.K * torch.softmax(logit_noise_est, dim=1, keepdim=True) - tau
            )
            return score
        else:
            # The model is predicting the score directly
            return self.net(x, t)

    def training_step(self, train_batch, _idxs) -> torch.Tensor:
        x, label = train_batch
        x = prob_to_logit(x)
        idx = torch.randint(
            self.num_scales,
            size=(x.shape[0],),
            device=self.device,
            dtype=torch.long,
            requires_grad=False,
        )
        tau = self.taus[idx][:, None, None, None]
        x_noisy = log_concrete_sample(x, tau=tau)
        scores = self.forward(x_noisy, idx.float())
        loss = categorical_dsm_loss(x, x_noisy, scores, tau)

        self.log("loss", loss.item())

        return loss

    def validation_step(self, val_batch, _idxs) -> torch.Tensor:

        val_loss = self.training_step(val_batch, None)
        self.log("val_loss", val_loss.item(), prog_bar=True)

        x, label = val_batch
        x = prob_to_logit(x)
        for idx in range(0, self.num_scales, self.num_scales // 5):
            t = torch.full(
                (x.shape[0],),
                idx,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            tau = self.taus[t.long()][:, None, None, None]
            x_noisy = log_concrete_sample(x, tau=tau)
            loss = categorical_dsm_loss(x, x_noisy, self.forward(x_noisy, t), tau)
            self.log(f"per_tau_loss/{tau[0].item():.1f}", loss.item())

        return val_loss
