import pdb
import functools
import ml_collections
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


from losses import categorical_dsm_loss
from models.mutils import (
    get_model,
    get_optimizer,
    get_taus,
    log_concrete_sample,
    onehot_to_logit,
    build_default_init_fn
)


class ScoreModel(pl.LightningModule):
    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()

        self.model_config = config.model
        for model_attr, val in config.model.to_dict().items():
            setattr(self, model_attr, val)

        self.net = get_model(config)
        self.training_opts = config.training
        self.optimization_opts = config.optim
        self.K = config.data.num_categories
        self.register_buffer("taus", torch.tensor(get_taus(config), device=self.device))

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

        ########! FIXME: This problem still exists ###############
        # I effectiveley CANNOT RESUME training is using Adam/AdamW
        # Should help with loading the step variable
        # as per https://github.com/pytorch/pytorch/issues/80809#issuecomment-1173481031
        # optimizer.param_groups[0]['capturable'] = True

        return optimizer

    def forward(self, x, t):

        if self.estimate_noise:
            """Assumes that the model is predicting the `logit noise`
            Following the log concrete gradient, we need to rescale
            the logits to turn them into scores
            """
            tau = self.taus[t.long()][:, None, None, None]
            logit_noise_est = self.net(x, t.float())
            score = tau * self.K * F.softmax(logit_noise_est, dim=1) - tau
            return score
        else:
            # The model is predicting the score directly
            return self.net(x, t)

    def training_step(self, train_batch, _idxs) -> torch.Tensor:

        x, label = train_batch
        x = onehot_to_logit(x)
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
        x = onehot_to_logit(x)
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

    def scorer(self, x_batch):
        pass


class TabScoreModel(pl.LightningModule):
    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()

        self.categories = config.data.categories
        self.continuous_dims = config.data.cont_dims
        self.categorical_dims = sum(self.categories)
        self.model_config = config.model

        for model_attr, val in config.model.to_dict().items():
            setattr(self, model_attr, val)

        self.net = get_model(config)
        self.training_opts = config.training
        self.optimization_opts = config.optim

        self.register_buffer("taus", torch.tensor(get_taus(config), device=self.device))

        # pdb.set_trace()
        default_init_fn = build_default_init_fn()
        self.net.apply(default_init_fn)

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

        ########! FIXME: This problem still exists ###############
        # I effectiveley CANNOT RESUME training is using Adam/AdamW
        # Should help with loading the step variable
        # as per https://github.com/pytorch/pytorch/issues/80809#issuecomment-1173481031
        # optimizer.param_groups[0]['capturable'] = True

        return optimizer

    def forward(self, x, t):
        # The model is predicting the score directly
        return self.net(x, t)

    def training_step(self, train_batch, _idxs) -> torch.Tensor:

        x, label = train_batch
        x_cont = x[:, : self.continuous_dims]
        x_cat = x[:, self.continuous_dims :]
        x_cat = onehot_to_logit(x_cat)

        idx = torch.randint(
            self.num_scales,
            size=(x.shape[0],),
            device=self.device,
            dtype=torch.long,
            requires_grad=False,
        )
        tau = self.taus[idx][:, None]
        splitter = functools.partial(
            torch.split, split_size_or_sections=self.categories, dim=1
        )

        x_noisy = torch.cat(
            [log_concrete_sample(x_cat_hot, tau=tau) for x_cat_hot in splitter(x_cat)],
            dim=1,
        )

        x_noisy = torch.cat((x_cont, x_noisy), dim=1)
        scores = self.forward(x_noisy, idx.float())

        loss = 0.0
        cat_scores = scores[:, self.continuous_dims :]
        x_cat_pert = x_noisy[:, self.continuous_dims :]
        for x_cat_logits, x_cat_noisy, x_cat_scores in zip(
            *map(splitter, [x_cat, x_cat_pert, cat_scores])
        ):
            loss += categorical_dsm_loss(x_cat_logits, x_cat_noisy, x_cat_scores, tau)

        self.log("loss", loss.item())

        return loss

    def validation_step(self, val_batch, _idxs) -> torch.Tensor:

        val_loss = self.training_step(val_batch, None)
        self.log("val_loss", val_loss.item(), prog_bar=True)

        return val_loss

    def scorer(self, x_batch):
        pass
