import functools
import pdb
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


class BaseScoreModel(pl.LightningModule):
    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()

        self.model_config = config.model
        for model_attr, val in config.model.to_dict().items():
            setattr(self, model_attr, val)

        assert self.embedding_type in ["fourier", "positional"]
        self.net = get_model(config)
        self.training_opts = config.training
        self.optimization_opts = config.optim

        # pdb.set_trace()
        if config.model.name != "ncsnpp":
            default_init_fn = build_default_init_fn()
            self.net.apply(default_init_fn)

        self.save_hyperparameters(config.to_dict())

    def on_load_checkpoint(self, checkpoint) -> None:
        """This is a hack to load the EMA weights into the model.
        The EMA weights are stored in the checkpoint as a callback.
        The EMA callback has a method called `replace_model_weights` that
        effectively loads the weights into the model.
        However, there is no way to access the callback from the model.
        So, we need to create an instance of the callback, load the weights
        into the callback, and then call the method to load the weights into
        the model.
        """
        # print(self.net.state_dict()['time_embed.0.W'])
        if "EMA" in checkpoint["callbacks"]:
            ema_callback = EMA(decay=0)
            ema_callback.load_state_dict(checkpoint["callbacks"]["EMA"])
            ema_callback.replace_model_weights(self)
            print(f"Restored EMA weights from checkpoint at step={checkpoint['global_step']}...")
            del ema_callback
        
        # as per https://github.com/pytorch/pytorch/issues/80809#issuecomment-1173481031
        checkpoint['optimizer_states'][0]['param_groups'][0]['capturable']=True
    
    def configure_callbacks(self):

        if self.ema_rate > 0.0:
            ema_callback = EMA(
                decay=0.999,
                evaluate_ema_weights_instead=True,
                save_ema_weights_in_callback_state=True,
            )
            return [ema_callback]

    def configure_optimizers(self):

        optim = get_optimizer(self.optimization_opts.optimizer)
        optimizer = optim(
            self.parameters(),
            lr=self.optimization_opts.lr,
            weight_decay=self.optimization_opts.weight_decay,
            betas=(self.optimization_opts.beta1, self.optimization_opts.beta2),
        )

        if self.optimization_opts.scheduler != "none":
            if self.optimization_opts.scheduler == "cycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=1e-3,
                    total_steps=self.training_opts.n_steps,
                    div_factor=1e2,  # starts at 1e-5
                    final_div_factor=1e-1,  # ends at 1e-6
                    three_phase=False,  # Triangle Rate
                    pct_start=0.01,
                    anneal_strategy="cos",
                )
            elif self.optimization_opts.scheduler == "step":  # Dfaults to StepLR
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer=optimizer,
                    step_size=int(0.4 * self.training_opts.n_steps),
                    gamma=0.3,
                )
            elif self.optimization_opts.scheduler == "cosine":  # Dfaults to StepLR
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.training_opts.n_steps,
                    eta_min=1e-5,
                )
            else:
                raise NotImplementedError("Scheduler not implemented")
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    # Using custom or multiple metrics (default_hp_metric=False)
    # def on_train_start(self):
    #     self.logger.log_hyperparams(self.hparams, {"val_err": 0})

    # def on_after_backward(self) -> None:
    #     if self.trainer.global_step % 10 == 0:
    #         for name, params in self.named_parameters():
    #             # print(name)
    #             self.logger.experiment.add_histogram(
    #                 f"{name}.act", params, self.current_epoch
    #             )
    #             if params.requires_grad:
    #                 self.logger.experiment.add_histogram(
    #                     f"{name}.grad", params.grad, self.current_epoch
    #                 )

    def forward(self, x, t):

        if self.estimate_noise:
            # The model is predicting the logit noise directly
            # Will be trained to only match the noise only via KL div loss etc.
            return self.net(x, t)

        # Predicting the scores
        return self.score_fn(x, t)

    def score_fn(self, x, t):
        pass

    def single_loss_step(self, x_batch, timestep_idxs):
        pass

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
        # x_batch = x_batch[label == 0]

        val_loss = 0.0
        val_err = 0.0

        t = torch.ones(
            size=(x_batch.shape[0],),
            device=self.device,
            dtype=torch.long,
            requires_grad=False,
        )
        L = self.num_scales

        for idx in range(0, self.num_scales, 3):
            cat_loss, cont_loss, rel_err = self.single_loss_step(x_batch, t * idx)
            loss = cat_loss + cont_loss
            val_loss += loss
            val_err += rel_err

            if idx % (L//3) == 0:
                self.log(f"val_loss/{idx}", loss.item())
                self.log(f"val_err/{idx}", rel_err.item())
        
        denom=len(range(0, self.num_scales, 3))
        val_loss /= denom
        val_err /= denom

        self.log("val_loss", val_loss.item(), prog_bar=True)
        self.log("val_err", val_err.item())

        return val_loss

    @torch.inference_mode()
    def scorer(self, x_batch, denoise_step=False):
        self.eval()
        N = self.num_scales
        score_norms = torch.zeros(x_batch.shape[0], N, device=x_batch.device)

        # Single denoising step
        if denoise_step:
            vec_t = torch.ones(x_batch.shape[0], device=x_batch.device) * (N - 1)
            score = self.score_fn(x_batch, vec_t)
            score *= torch.linalg.norm(score.reshape(score.shape[0], -1), dim=1)[
                :, None, None, None
            ]
            if self.continuous_channels > 0:
                x_cont = x_batch[:, : self.continuous_channels]
                x_cat = x_batch[:, self.continuous_channels :]
                x_cat += score
                x_cat -= torch.logsumexp(x_cat, dim=1, keepdim=True)
                # Cont channels only used as conditioning information
                x_batch = torch.cat([x_cont, x_cat], dim=1)
            else:
                x_batch += score
                x_batch -= torch.logsumexp(x_batch, dim=1, keepdim=True)

        for idx in range(N):
            vec_t = (
                torch.ones((x_batch.shape[0],), device=x_batch.device, dtype=torch.long)
                * idx
            )
            # x_batch = smoother(x_batch, self.taus[vec_t.long()][:, None, None, None])
            score = self.score_fn(x_batch, vec_t)
            score_norms[:, idx] = torch.linalg.norm(
                score.reshape(score.shape[0], -1), dim=1
            )

        return score_norms


class VisionScoreModel(BaseScoreModel):
    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__(config)

        self.K = config.data.categorical_channels
        self.continuous_channels = config.data.continuous_channels
        self.register_buffer("taus", torch.tensor(get_taus(config), device=self.device))
        if config.model.estimate_noise:
            self.loss_fn = KL_loss
        else:
            self.loss_fn = categorical_dsm_loss

    def score_fn(self, x, t):
        tau = self.taus[t.long()][:, None, None, None]
        logit_noise_est = self.net(x, t.float())
        # print(logit_noise_est.mean())
        # pdb.set_trace()
        score = tau * self.K * F.softmax(logit_noise_est, dim=1) - tau
        return score

    def single_loss_step(self, x_batch, timestep_idxs):

        # print("before noise", x_batch.mean(), x_batch.std())
        tau = self.taus[timestep_idxs][:, None, None, None]
        # print(tau)
        if self.continuous_channels > 0:
            x_cat = x_batch[:, self.continuous_channels :]
            x_cont = x_batch[:, : self.continuous_channels]

            x_cat_noisy = log_concrete_sample(x_cat, tau=tau)
            # Cont channels used as conditioning information
            x_input = torch.cat([x_cont, x_cat_noisy], dim=1)
        else:
            x_cat = x_batch
            x_cat_noisy = x_input = log_concrete_sample(x_cat, tau=tau)
        # print(x_cat_noisy.mean(), x_cat_noisy.std())
        scores = self.forward(x_input, timestep_idxs.float())
        cat_loss, rel_err = self.loss_fn(x_cat, x_cat_noisy, scores, tau)
        # print(cat_loss, rel_err)
        cont_loss = 0.0

        return cat_loss, cont_loss, rel_err


class TabScoreModel(BaseScoreModel):
    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__(config)

        self.categories = config.data.categories
        self.continuous_dims = config.data.numerical_features
        self.categorical_dims = sum(self.categories)

        self.register_buffer("sigmas", torch.tensor(get_sigmas(config)))
        self.register_buffer("taus", torch.tensor(get_taus(config), device=self.device))

        self.splitter = functools.partial(
            torch.split, split_size_or_sections=self.categories, dim=1
        )

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
        # print(cont_score.shape, cat_score.shape, score.shape)
        return score

    def single_loss_step(self, x_batch, timestep_idxs):

        tau = self.taus[timestep_idxs][:, None]
        sigma = self.sigmas[timestep_idxs][:, None]

        x_cont = x_batch[:, : self.continuous_dims]
        x_cat = x_batch[:, self.continuous_dims :]
        # x_cat = onehot_to_logit(x_cat)

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
            l, err = continuous_dsm_loss(cont_noise, cont_scores, sigma)
            cont_loss += l
            rel_err += err

        return cat_loss, cont_loss, rel_err

    def validation_step(self, val_batch, batch_idxs) -> torch.Tensor:
        x_batch, label = val_batch
        # print(label)
        x_batch = x_batch[label == 0]
        # idxs = torch.randint(
        #     self.num_scales,
        #     size=(x.shape[0],),
        #     device=self.device,
        #     dtype=torch.long,
        #     requires_grad=False,
        # )
        # val_loss = self.single_loss_step((x_batch, None), idxs)
        # self.log("val_loss", val_loss.item(), prog_bar=True)

        val_loss_cont = 0.0
        val_loss_cat = 0.0
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
            val_loss_cont += cont_loss
            val_loss_cat += cat_loss
            val_err += rel_err

            if idx % 3 == 0:
                self.log(f"val_loss/{idx}", loss.item())
                self.log(f"val_err/{idx}", rel_err.item())

        val_loss_cont /= self.num_scales
        val_loss_cat /= self.num_scales
        val_err /= self.num_scales
        val_loss = val_loss_cont + val_loss_cat

        self.log("val_loss", val_loss.item(), prog_bar=True)
        self.log("val_err", val_err.item())
        self.log("val_loss_cat", val_loss_cat.item())

        if val_loss_cont > 0:
            self.log("val_loss_cont", val_loss_cont.item())

        return val_loss

    @torch.inference_mode()
    def scorer(self, x_batch, denoise_step=False):
        self.eval()
        N = self.num_scales
        score_norms = torch.zeros(x_batch.shape[0], N, device=x_batch.device)

        x_cont = x_batch[:, : self.continuous_dims]
        x_categories = x_batch[:, self.continuous_dims :]
        x_batch = torch.cat((x_cont, x_categories), dim=1).cuda()

        # Single denoising step
        if denoise_step:
            vec_t = torch.ones(x_batch.shape[0], device=x_batch.device) * (N - 1)
            score = self.score_fn(x_batch, vec_t)
            cont_scores = score[:, : self.continuous_dims]
            cont_scores *= torch.linalg.norm(cont_scores, dim=1, keepdim=True) ** -1
            cat_scores = score[:, self.continuous_dims :]
            cat_scores = torch.cat(
                [
                    x_cat_score
                    * (torch.linalg.norm(x_cat_score, dim=1, keepdim=True) ** -1)
                    for x_cat_score in self.splitter(cat_scores)
                ],
                dim=1,
            )
            score = torch.cat((cont_scores, cat_scores), dim=1)
            x_batch += score
            for x_cat in self.splitter(x_batch[:, self.continuous_dims :]):
                x_cat -= torch.logsumexp(x_cat, dim=1, keepdim=True)

        x_cont = x_batch[:, : self.continuous_dims]
        x_categories = x_batch[:, self.continuous_dims :]

        for idx in range(N):
            vec_t = (
                torch.ones((x_batch.shape[0],), device=x_batch.device, dtype=torch.long)
                * idx
            )
            tau = self.taus[vec_t.long()][:, None]

            # x_cat = torch.cat(
            #     [
            #         smoother(x_cat_hot, tau=tau)
            #         for x_cat_hot in self.splitter(x_categories)
            #     ],
            #     dim=1,
            # )
            # x_batch = torch.cat((x_cont, x_cat), dim=1).cuda()

            score = self.score_fn(x_batch, vec_t)
            score_norms[:, idx] = torch.linalg.norm(
                score.reshape(score.shape[0], -1), dim=1
            )

        return score_norms


def smoother(class_logits: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:

    eps = 1e-20
    U = torch.rand_like(class_logits)
    epsilon_noise = -torch.log(-torch.log(U + eps) + eps)
    epsilon_noise *= 1e-10

    x = (epsilon_noise + class_logits) / tau
    x = x - torch.logsumexp(x, dim=1, keepdim=True)

    return x
