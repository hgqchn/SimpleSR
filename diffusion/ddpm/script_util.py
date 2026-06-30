"""DDPM 超分 baseline 构建工具。

该文件替代 guided-diffusion 原始的脚本参数工具：
- 删除 classifier 相关配置。
- 删除 fp16 配置。
- 只保留超分模型 `SuperResModel` 和 DDPM/Spaced DDPM diffusion 构建。

guided-diffusion 官方超分采样脚本中常见配置：
- 64 -> 256 upsampler:
  `large_size=256, small_size=64, num_channels=192, num_res_blocks=2,
  learn_sigma=True, class_cond=False, noise_schedule=linear`
- 128 -> 512 upsampler:
  `large_size=512, small_size=128, num_channels=192, num_res_blocks=2,
  learn_sigma=True, class_cond=True, noise_schedule=linear`
- 采样时常配合 `timestep_respacing=250` 或类似少步设置。

本项目当前精简版不保留 `learn_sigma` 和 `class_cond`，因此 baseline 默认：
`out_channels=3, objective=pred_noise, fixed variance`。
"""

from __future__ import annotations

from .gaussian_diffusion import GaussianDiffusionSR, get_named_beta_schedule
from .respace import SpacedDiffusionSR, space_timesteps
from .unet import SuperResModel


def sr_model_defaults() -> dict:
    """SimpleSR DDPM 超分 UNet 默认配置。"""
    return {
        "large_size": 256,
        "small_size": 64,
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 2,
        "attention_resolutions": (32, 16, 8),
        "channel_mult": (1, 1, 2, 2, 4, 4),
        "dropout": 0.0,
        "num_heads": 4,
    }


def diffusion_defaults() -> dict:
    """DDPM 超分 diffusion 默认配置。"""
    return {
        "diffusion_steps": 1000,
        "noise_schedule": "linear",
        "timestep_respacing": "",
        "objective": "pred_noise",
        "clip_denoised": True,
    }


def create_sr_model(
    large_size: int,
    in_channels: int = 3,
    model_channels: int = 192,
    out_channels: int = 3,
    num_res_blocks: int = 2,
    attention_resolutions: tuple[int, ...] = (32, 16, 8),
    channel_mult: tuple[int, ...] = (1, 1, 2, 2, 4, 4),
    dropout: float = 0.0,
    num_heads: int = 4,
) -> SuperResModel:
    """创建超分 UNet。

    `small_size` 不参与网络结构构建，低分图像会在 forward 中上采样到 HR 尺寸。
    """
    return SuperResModel(
        image_size=large_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
        dropout=dropout,
        num_heads=num_heads,
    )


def create_sr_diffusion(
    diffusion_steps: int = 1000,
    noise_schedule: str = "linear",
    timestep_respacing="",
    objective: str = "pred_noise",
    clip_denoised: bool = True,
    scale: int = 4,
):
    """创建超分 DDPM diffusion。

    如果 `timestep_respacing` 为空，返回完整步数 `GaussianDiffusionSR`。
    否则返回少步采样版本 `SpacedDiffusionSR`。
    """
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    kwargs = {
        "betas": betas,
        "objective": objective,
        "clip_denoised": clip_denoised,
        "scale": scale,
    }
    if timestep_respacing:
        return SpacedDiffusionSR(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            **kwargs,
        )
    return GaussianDiffusionSR(**kwargs)


def create_sr_model_and_diffusion(**kwargs):
    """同时创建超分 UNet 和 diffusion。

    该函数方便做 baseline 实验；在 SimpleSR 正式训练流程中，也可以分别通过
    `build_network` 和 `build_diffusion` 创建。
    """
    model_cfg = sr_model_defaults()
    diffusion_cfg = diffusion_defaults()

    for key in list(model_cfg):
        if key in kwargs:
            model_cfg[key] = kwargs.pop(key)
    for key in list(diffusion_cfg):
        if key in kwargs:
            diffusion_cfg[key] = kwargs.pop(key)
    if "scale" in kwargs:
        diffusion_cfg["scale"] = kwargs.pop("scale")
    if kwargs:
        raise ValueError(f"Unknown DDPM config keys: {sorted(kwargs)}")

    # small_size 只用于记录官方配置含义，不传给模型。
    model_cfg.pop("small_size", None)
    model = create_sr_model(**model_cfg)
    diffusion = create_sr_diffusion(**diffusion_cfg)
    return model, diffusion
