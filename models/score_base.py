import pdb
import functools
import ml_collections
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


from losses import KL_loss, categorical_dsm_loss, continuous_dsm_loss, log_concrete_grad
from models.mutils import (
    get_model,
    get_optimizer,
    get_taus,
    get_sigmas,
    log_concrete_sample,
    onehot_to_logit,
    build_default_init_fn,
)

from models.ema import EMA


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

        self.register_buffer("sigmas", torch.tensor(get_sigmas(config)))
        self.register_buffer("taus", torch.tensor(get_taus(config), device=self.device))

        self.splitter = functools.partial(
            torch.split, split_size_or_sections=self.categories, dim=1
        )

        # pdb.set_trace()
        default_init_fn = build_default_init_fn()
        self.net.apply(default_init_fn)

    def on_load_checkpoint(self, checkpoint) -> None:
        '''This is a hack to load the EMA weights into the model. 
        The EMA weights are stored in the checkpoint as a callback.
        The EMA callback has a method called `replace_model_weights` that
        effectively loads the weights into the model. 
        However, there is no way to access the callback from the model.
        So, we need to create an instance of the callback, load the weights
        into the callback, and then call the method to load the weights into
        the model.
        '''
        # print(self.net.state_dict()['time_embed.0.W'])
        if "EMA" in checkpoint["callbacks"]:
            ema_callback = EMA(decay=0)
            ema_callback.load_state_dict(checkpoint["callbacks"]["EMA"])
            ema_callback.replace_model_weights(self)
            print("Restored checkpoint from EMA...")
            del ema_callback

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
            # The model is predicting the logit noise directly
            # Will be trained to only match the noise only via KL div loss etc.
            return self.net(x, t)

        # Predicting the scores
        return self.score_fn(x, t)

    def score_fn(self, x, t):
        """Assumes that the model is predicting the `logit noise`
        Following the log concrete gradient, we need to rescale
        the logits to turn them into scores
        """
        tau = self.taus[t.long()][:, None]
        logit_noise_est = self.net(x, t.float())
        cat_logits = logit_noise_est[:, self.continuous_dims :]
        cont_score = logit_noise_est[:, : self.continuous_dims]

        cat_score = torch.cat(
            [l.shape[1] * F.softmax(l, dim=1) for l in self.splitter(cat_logits)],
            dim=1,
        )
        cat_score = tau * cat_score - tau

        score = torch.cat((cont_score, cat_score), dim=1)

        return score

    def single_loss_step(self, x_batch, timestep_idxs):

        tau = self.taus[timestep_idxs][:, None]
        sigma = self.sigmas[timestep_idxs][:, None]

        x_cont = x_batch[:, : self.continuous_dims]
        x_cat = x_batch[:, self.continuous_dims :]
        x_cat = onehot_to_logit(x_cat)

        # Add noise appropriately
        x_noisy = torch.cat(
            [
                log_concrete_sample(x_cat_hot, tau=tau)
                for x_cat_hot in self.splitter(x_cat)
            ],
            dim=1,
        )

        if x_cont.shape[1] > 0:
            # g=torch.Generator(device=x_cont.device)
            # g.manual_seed(42)
            # cont_noise = torch.randn(x_cont.shape, generator=g, device=x_cont.device) * sigma
            cont_noise = torch.randn_like(x_cont) * sigma
            x_cont_noisy = x_cont + cont_noise
            x_noisy = torch.cat((x_cont_noisy, x_noisy), dim=1)

        scores = self.forward(x_noisy, timestep_idxs.float())

        # Compute losses
        cat_loss = 0.0
        cont_loss = 0.0
        rel_err = 0.0

        # Categorical loss
        cat_scores = scores[:, self.continuous_dims :]
        x_cat_pert = x_noisy[:, self.continuous_dims :]
        for i, (x_cat_logits, x_cat_noisy, x_cat_scores) in enumerate(
            zip(*map(self.splitter, [x_cat, x_cat_pert, cat_scores]))
        ):
            # pdb()
            if self.estimate_noise:
                l, err = KL_loss(x_cat_logits, x_cat_noisy, x_cat_scores, tau)
            else:
                l, err = categorical_dsm_loss(
                    x_cat_logits, x_cat_noisy, x_cat_scores, tau
                )
            cat_loss += l
            rel_err += err
            # self.log(f"loss/cat_idx={i}", l.item())
            # print(categorical_dsm_loss(x_cat_logits, x_cat_noisy, x_cat_scores, tau))

        # Continuous loss
        if x_cont.shape[1] > 0:
            cont_scores = scores[:, : self.continuous_dims]
            cont_loss += continuous_dsm_loss(cont_noise, cont_scores, sigma)

        return cat_loss, cont_loss, rel_err

    def training_step(self, train_batch, batch_idxs) -> torch.Tensor:

        x, label = train_batch

        idxs = torch.randint(
            self.num_scales,
            size=(x.shape[0],),
            device=self.device,
            dtype=torch.long,
            requires_grad=False,
        )

        cat_loss, cont_loss, rel_err = self.single_loss_step(x, idxs)

        loss = cat_loss + cont_loss

        # if self.training:
        self.log("loss", loss.item())
        if cat_loss > 0:
            self.log("loss/cat", cat_loss.item())
        if cont_loss > 0:
            self.log("loss/cont", cont_loss.item())

        self.log("rel_err", rel_err.item())

        return loss

    def validation_step(self, val_batch, batch_idxs) -> torch.Tensor:
        x_batch, label = val_batch
        # print(label, label.argmax(dim=1) == 0)
        x_batch = x_batch[label.argmax(dim=1) == 0]
        # idxs = torch.randint(
        #     self.num_scales,
        #     size=(x.shape[0],),
        #     device=self.device,
        #     dtype=torch.long,
        #     requires_grad=False,
        # )
        # val_loss = self.single_loss_step((x_batch, None), idxs)
        # self.log("val_loss", val_loss.item(), prog_bar=True)

        val_loss = 0.0
        val_err = 0.0

        t = torch.ones(
            size=(x_batch.shape[0],),
            device=self.device,
            dtype=torch.long,
            requires_grad=False,
        )

        # timesteps_to_log = 3
        for idx in range(0, self.num_scales):
            cat_loss, cont_loss, rel_err = self.single_loss_step(x_batch, t * idx)
            loss = cat_loss + cont_loss
            val_loss += loss
            val_err += rel_err

            if idx % 3 == 0:
                self.log(f"val_loss/{idx}", loss.item())
                self.log(f"val_err/{idx}", rel_err.item())

        val_loss /= self.num_scales
        val_err /= self.num_scales
        
        self.log("val_loss", val_loss.item(), prog_bar=True)
        self.log("val_err", val_err.item())

        return val_loss

    @torch.no_grad()
    def scorer(self, x_batch, denoise_step=False):
        self.eval()
        N = self.num_scales
        score_norms = torch.empty(N, x_batch.shape[0], device=x_batch.device)
        
        # Single denoising step
        if denoise_step:
            vec_t = torch.ones(x_batch.shape[0], device=x_batch.device) * (N-1)
            score = self.score_fn(x_batch, vec_t) * 1e-2
            x_batch += score
            x_batch -= torch.logsumexp(x_batch, dim=1, keepdim=True)
            
        for idx in range(N):
            vec_t = torch.ones((x_batch.shape[0],), device=x_batch.device, dtype=torch.long) * idx
            score = self.score_fn(x_batch, vec_t)
            score_norms[idx, :] = torch.linalg.norm(score.reshape(score.shape[0],-1), dim=1)
        
        return score_norms