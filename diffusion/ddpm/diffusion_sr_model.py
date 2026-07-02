import torch

from diffusion.ddpm.resample import create_named_schedule_sampler
from simplesr.models.base_sr_model import SRModel
from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.object_utils import instantiate_from_config


def ddpm_training_step(model, diffusion, x_start, lq=None, schedule_sampler=None):
    sampler = schedule_sampler or create_named_schedule_sampler("uniform", diffusion)
    timesteps, _ = sampler.sample(x_start.shape[0], x_start.device)

    if lq is None:
        terms = diffusion.training_losses(model, x_start, t=timesteps)
    else:
        terms = diffusion.training_losses(model, x_start, lq, t=timesteps)

    loss = terms["loss"].mean()
    log_dict = {"loss": loss.detach()}
    if "mse" in terms:
        log_dict["mse"] = terms["mse"].mean().detach()
    log_dict = {k: v.detach() if torch.is_tensor(v) else v for k, v in log_dict.items()}
    return loss, log_dict


class DiffusionSRModel(SRModel):
    """SimpleSR wrapper for DDPM super-resolution."""

    def __init__(self, opt):
        diffusion_opt = opt["diffusion"]
        if "train" in diffusion_opt:
            self.diffusion = instantiate_from_config(diffusion_opt["train"])
            self.sample_diffusion = instantiate_from_config(diffusion_opt["val"])
            self.diffusion_train_opt = diffusion_opt.get("train", {})
            self.diffusion_val_opt = diffusion_opt.get("val", {})
        else:
            self.diffusion = instantiate_from_config(diffusion_opt)
            self.sample_diffusion = self.diffusion
            self.diffusion_train_opt = diffusion_opt
            self.diffusion_val_opt = {}
        super().__init__(opt)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]
        self.schedule_sampler = create_named_schedule_sampler(
            self.diffusion_train_opt.get("schedule_sampler", "uniform"), self.diffusion
        )

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"使用 EMA，decay={self.ema_decay}")
            self.net_g_ema = self.init_ema(self.net_g, decay=self.ema_decay)
            if self.load_path is not None:
                self.load_network(self.net_g_ema.module, self.load_path, strict=True)

        self.setup_optimizers()
        self.setup_schedulers()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        loss, loss_dict = ddpm_training_step(
            self.net_g,
            self.diffusion,
            self.gt,
            lq=self.lq,
            schedule_sampler=self.schedule_sampler,
        )
        loss.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.update_ema(self.net_g, self.net_g_ema)

    def test(self):
        sampler = self.diffusion_val_opt.get("sampler", "ddpm")
        if hasattr(self, "net_g_ema"):
            net_g = self.get_ema_module(self.net_g_ema)
            net_g.eval()
            with torch.no_grad():
                self.output = self.sample_diffusion.sample(net_g, self.lq, sampler=sampler)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.sample_diffusion.sample(self.net_g, self.lq, sampler=sampler)
            self.net_g.train()
