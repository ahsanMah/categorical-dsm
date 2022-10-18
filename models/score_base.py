from abc import abstractmethod

import ml_collections
import pytorch_lightning as pl
import torch
from losses import categorical_dsm_loss

from mutils import get_taus, log_concrete_sample, prob_to_logit


class BaseScoreNet(pl.LightningModule):
    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()

        self.model_config = config.model
        for model_attr, val in config.model.to_dict().items():
            setattr(self, model_attr, val)

        self.training_opts = config.training
        self.register_buffer("taus", torch.tensor(get_taus(config)))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.training_opts.lr)

        if self.training_opts.scheduler is not "skip":
            scheduler = torch.lr_scheduler.StepLR(
                step_size=int(0.3 * self.training_opts.n_iters), gamma=0.3
            )
            return optimizer, scheduler

        return optimizer

    @abstractmethod
    def forward(self, x, t) -> torch.Tensor:
        pass

    def training_step(self, train_batch) -> torch.Tensor:
        x, label = train_batch

        x = prob_to_logit(x)
        idx = torch.randint(self.num_scales, size=(x.shape[0],), dtype=torch.long)
        tau = self.taus[idx][:, None, None, None]
        x_noisy = log_concrete_sample(x, tau=tau)
        t = torch.ones(x.shape[0]) * idx

        scores = self.forward(x_noisy, t)

        loss = categorical_dsm_loss(x, x_noisy, scores, tau)
        
        self.log('loss', loss.item())

        return loss

    def validation_step(self, val_batch) -> torch.Tensor:
        x, label = val_batch

        x = prob_to_logit(x)
        idx = torch.randint(self.num_scales, size=(x.shape[0],), dtype=torch.long)
        tau = self.taus[idx][:, None, None, None]
        x_noisy = log_concrete_sample(x, tau=tau)
        t = torch.ones(x.shape[0]) * idx
        scores = self.forward(x_noisy, t)
        loss = categorical_dsm_loss(x, x_noisy, scores, tau)

        self.log('val_loss', loss.item())

        return loss