"""Simplified DDPM/DDIM diffusion for super-resolution.

精简版 DDPM/DDIM 扩散过程，用于超分辨率训练和采样。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from diffusion.common_utils import extract_into_tensor, get_named_beta_schedule, mean_flat


class GaussianDiffusion:
    """Base DDPM/DDIM diffusion.

    普通 DDPM/DDIM 扩散类。
    """

    def __init__(
        self,
        betas: np.ndarray | None = None,
        diffusion_steps: int = 1000,
        noise_schedule: str = "linear",
        objective: str = "pred_noise",
    ) -> None:
        """Initialize diffusion schedules.

        初始化扩散 beta schedule 和所有预计算系数。
        """
        if objective not in {"pred_noise", "pred_x0"}:
            raise ValueError(f"Unsupported objective: {objective}")

        self.objective = objective

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

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from q(x_t | x_0).

        从前向扩散分布 q(x_t | x_0) 采样。
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_xstart_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Convert predicted noise to predicted x_start.

        根据预测噪声反推出预测的干净图像 x_start。
        """
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_xstart(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_start: torch.Tensor,
    ) -> torch.Tensor:
        """Convert predicted x_start to predicted noise.

        根据预测的干净图像 x_start 反推出噪声。
        """
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute q(x_{t-1} | x_t, x_0).

        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值、方差和 log 方差。
        """
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        log_var = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def call_model(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Call denoising model.

        调用去噪模型；子类可重写以注入条件。
        """
        return model(x_t, t, **model_kwargs)

    def model_predictions(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict both noise and x_start.

        统一得到模型预测噪声和预测 x_start，提升采样函数可读性。
        """
        if model_kwargs is None:
            model_kwargs = {}

        model_out = self.call_model(model, x, t, **model_kwargs)
        if self.objective == "pred_noise":
            pred_noise = model_out
            pred_xstart = self.predict_xstart_from_noise(x, t, pred_noise)
        elif self.objective == "pred_x0":
            pred_xstart = model_out
            pred_noise = self.predict_noise_from_xstart(x, t, pred_xstart)
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")

        if denoised_fn is not None:
            pred_xstart = denoised_fn(pred_xstart)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
            pred_noise = self.predict_noise_from_xstart(x, t, pred_xstart)
        return {"pred_noise": pred_noise, "pred_xstart": pred_xstart}

    def p_mean_variance(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute p(x_{t-1} | x_t) parameters.

        计算反向一步分布 p(x_{t-1} | x_t) 的均值、方差和预测 x_start。
        """
        preds = self.model_predictions(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        mean, var, log_var = self.q_posterior_mean_variance(preds["pred_xstart"], x, t)
        return {
            "mean": mean,
            "variance": var,
            "log_variance": log_var,
            "pred_xstart": preds["pred_xstart"],
        }

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[..., torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample one DDPM reverse step.

        采样 DDPM 反向过程的一步。
        """
        if cond_fn is not None:
            raise NotImplementedError("cond_fn is not kept in this simplified DDPM.")
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...] | list[int],
        noise: torch.Tensor | None = None,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[..., torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        device: torch.device | None = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Run full DDPM sampling loop.

        从纯噪声开始执行完整 DDPM 采样循环。
        """
        if device is None:
            device = next(model.parameters()).device
        img = noise if noise is not None else torch.randn(*shape, device=device)
        indices = reversed(range(self.num_timesteps))
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = torch.full((shape[0],), i, device=img.device, dtype=torch.long)
            img = self.p_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )["sample"]
        return img

    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[..., torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        eta: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Sample one DDIM reverse step.

        采样 DDIM 反向过程的一步。
        """
        if cond_fn is not None:
            raise NotImplementedError("cond_fn is not kept in this simplified DDPM.")
        preds = self.model_predictions(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        mean_pred = (
            preds["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * preds["pred_noise"]
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": preds["pred_xstart"]}

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...] | list[int],
        noise: torch.Tensor | None = None,
        clip_denoised: bool = True,
        denoised_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cond_fn: Callable[..., torch.Tensor] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        device: torch.device | None = None,
        progress: bool = False,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Run full DDIM sampling loop.

        从纯噪声开始执行完整 DDIM 采样循环。
        """
        if device is None:
            device = next(model.parameters()).device
        img = noise if noise is not None else torch.randn(*shape, device=device)
        indices = reversed(range(self.num_timesteps))
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = torch.full((shape[0],), i, device=img.device, dtype=torch.long)
            img = self.ddim_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )["sample"]
        return img

    def training_losses(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: dict[str, Any] | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss terms.

        计算训练 loss 项，返回 ADM 风格的 terms 字典。
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise)
        model_out = self.call_model(model, x_t, t, **model_kwargs)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")

        mse = mean_flat((model_out - target) ** 2)
        return {"loss": mse, "mse": mse.detach()}

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...] | list[int],
        noise: torch.Tensor | None = None,
        sampler: str = "ddpm",
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Dispatch sampling by sampler name.

        根据 sampler 名称分发到 DDPM 或 DDIM 采样。
        """
        if sampler == "ddim":
            return self.ddim_sample_loop(model, shape, noise=noise, model_kwargs=model_kwargs, **kwargs)
        if sampler == "ddpm":
            return self.p_sample_loop(model, shape, noise=noise, model_kwargs=model_kwargs, **kwargs)
        raise ValueError(f"Unknown sampler: {sampler}")


class GaussianDiffusionSR(GaussianDiffusion):
    """Super-resolution DDPM/DDIM diffusion.

    超分辨率 DDPM/DDIM 扩散类。
    """

    def __init__(self, *args: Any, scale: int = 4, **kwargs: Any) -> None:
        """Initialize SR diffusion.

        初始化超分扩散类，并记录放大倍率。
        """
        super().__init__(*args, **kwargs)
        self.scale = scale

    def call_model(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Call SR denoising model with low-quality condition.

        调用带低分辨率条件的超分去噪模型。
        """
        lq = model_kwargs.pop("lq")
        return model(x_t, t, lq=lq, **model_kwargs)

    def training_losses(
        self,
        model: torch.nn.Module,
        gt: torch.Tensor,
        lq: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: dict[str, Any] | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute SR diffusion training loss terms.

        计算超分扩散训练 loss 项。
        """
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        model_kwargs["lq"] = lq
        return super().training_losses(model, gt, t=t, model_kwargs=model_kwargs, noise=noise)

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        lq: torch.Tensor,
        shape: tuple[int, ...] | None = None,
        noise: torch.Tensor | None = None,
        sampler: str = "ddpm",
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sample super-resolved images from low-quality inputs.

        根据低分辨率输入采样超分辨率图像。
        """
        if shape is None:
            shape = (lq.shape[0], 3, lq.shape[2] * self.scale, lq.shape[3] * self.scale)
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        model_kwargs["lq"] = lq
        if sampler == "ddim":
            return self.ddim_sample_loop(model, shape, noise=noise, model_kwargs=model_kwargs, **kwargs)
        if sampler == "ddpm":
            return self.p_sample_loop(model, shape, noise=noise, model_kwargs=model_kwargs, **kwargs)
        raise ValueError(f"Unknown sampler: {sampler}")
