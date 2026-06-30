"""精简 DDPM / DDIM 扩散过程。


"""

from __future__ import annotations

import math

import numpy as np
import torch




class GaussianDiffusion:
    """普通 DDPM/DDIM 扩散过程。

    Args:
        betas: shape 为 ``(T,)`` 的 beta schedule。
        objective: 模型预测目标，支持 ``pred_noise``、``pred_x0``、``pred_v``。
        clip_denoised: 采样时是否把预测 x0 clamp 到 [-1, 1]。
    """

    def __init__(
        self,
        betas: np.ndarray | None = None,
        diffusion_steps: int = 1000,
        noise_schedule: str = "linear",
        objective: str = "pred_noise",
        clip_denoised: bool = True,
    ):
        if objective not in {"pred_noise", "pred_x0", "pred_v"}:
            raise ValueError(f"Unsupported objective: {objective}")

        self.objective = objective
        self.clip_denoised = clip_denoised

        if betas is None:
            betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        betas = np.array(betas, dtype=np.float64)
        if betas.ndim != 1:
            raise ValueError("betas must be a 1-D array")
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError("betas must be in (0, 1]")

        self.betas = betas
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """从 q(x_t | x_0) 采样。"""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_xstart_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_xstart(self, x_t: torch.Tensor, t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_xstart_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 q(x_{t-1} | x_t, x_0)。"""
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        log_var = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def call_model(self, model, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs) -> torch.Tensor:
        """模型调用入口，子类可覆盖以注入条件。"""
        return model(x_t, t, **model_kwargs)

    def model_predictions(self, model, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs) -> ModelPrediction:
        model_out = self.call_model(model, x_t, t, **model_kwargs)
        if self.objective == "pred_noise":
            pred_noise = model_out
            pred_xstart = self.predict_xstart_from_noise(x_t, t, pred_noise)
        elif self.objective == "pred_x0":
            pred_xstart = model_out
            pred_noise = self.predict_noise_from_xstart(x_t, t, pred_xstart)
        else:
            pred_xstart = self.predict_xstart_from_v(x_t, t, model_out)
            pred_noise = self.predict_noise_from_xstart(x_t, t, pred_xstart)

        if self.clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        return ModelPrediction(pred_noise=pred_noise, pred_xstart=pred_xstart)

    def p_mean_variance(self, model, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs) -> dict:
        preds = self.model_predictions(model, x_t, t, **model_kwargs)
        mean, var, log_var = self.q_posterior_mean_variance(preds.pred_xstart, x_t, t)
        return {
            "mean": mean,
            "variance": var,
            "log_variance": log_var,
            "pred_xstart": preds.pred_xstart,
        }

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs) -> dict:
        out = self.p_mean_variance(model, x_t, t, **model_kwargs)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    @torch.no_grad()
    def p_sample_loop(self, model, shape: tuple[int, ...], noise: torch.Tensor | None = None, **model_kwargs):
        img = noise if noise is not None else torch.randn(shape, device=next(model.parameters()).device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=img.device, dtype=torch.long)
            img = self.p_sample(model, img, t, **model_kwargs)["sample"]
        return img

    @torch.no_grad()
    def ddim_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, eta: float = 0.0, **model_kwargs) -> dict:
        preds = self.model_predictions(model, x_t, t, **model_kwargs)
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        mean_pred = (
            preds.pred_xstart * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * preds.pred_noise
        )
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": preds.pred_xstart}

    @torch.no_grad()
    def ddim_sample_loop(
        self, model, shape: tuple[int, ...], noise: torch.Tensor | None = None, eta: float = 0.0, **model_kwargs
    ):
        img = noise if noise is not None else torch.randn(shape, device=next(model.parameters()).device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=img.device, dtype=torch.long)
            img = self.ddim_sample(model, img, t, eta=eta, **model_kwargs)["sample"]
        return img

    def training_losses(
        self,
        model,
        x_start: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """计算 DDPM 训练损失。返回总 loss 和日志字典。"""
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise)
        model_out = self.call_model(model, x_t, t, **model_kwargs)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            target = self.predict_v(x_start, t, noise)

        mse = mean_flat((model_out - target) ** 2)
        loss = mse.mean()
        return loss, {"loss": loss.detach(), "mse": mse.mean().detach()}

    @torch.no_grad()
    def sample(self, model, shape: tuple[int, ...], noise: torch.Tensor | None = None, sampler: str = "ddpm"):
        if sampler == "ddim":
            return self.ddim_sample_loop(model, shape, noise=noise)
        if sampler == "ddpm":
            return self.p_sample_loop(model, shape, noise=noise)
        raise ValueError(f"Unknown sampler: {sampler}")


class GaussianDiffusionSR(GaussianDiffusion):
    """超分 DDPM。

    低分图像条件由模型自身处理：默认要求模型 forward 签名为
    ``model(x_t, t, lq=...)``，例如 `SuperResModel`。
    """

    def __init__(self, *args, scale: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def call_model(self, model, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs) -> torch.Tensor:
        lq = model_kwargs.pop("lq")
        return model(x_t, t, lq=lq, **model_kwargs)

    def training_losses(
        self,
        model,
        gt: torch.Tensor,
        lq: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        return super().training_losses(model, gt, t=t, noise=noise, lq=lq)

    @torch.no_grad()
    def sample(
        self,
        model,
        lq: torch.Tensor,
        shape: tuple[int, ...] | None = None,
        noise: torch.Tensor | None = None,
        sampler: str = "ddpm",
    ) -> torch.Tensor:
        if shape is None:
            shape = (lq.shape[0], 3, lq.shape[2] * self.scale, lq.shape[3] * self.scale)
        if sampler == "ddim":
            return self.ddim_sample_loop(model, shape, noise=noise, lq=lq)
        if sampler == "ddpm":
            return self.p_sample_loop(model, shape, noise=noise, lq=lq)
        raise ValueError(f"Unknown sampler: {sampler}")
