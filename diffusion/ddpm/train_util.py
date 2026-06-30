"""DDPM 训练辅助函数。

guided-diffusion 原始 `TrainLoop` 已删除。SimpleSR 使用自己的 `BaseModel`
和 `train.py` 管理训练、DDP、EMA、checkpoint、日志和验证。
"""

import torch

from .resample import UniformSampler


def ddpm_training_step(model, diffusion, x_start, lq=None, schedule_sampler=None):
    """计算一次 DDPM/SR-DDPM 训练 loss。

    Args:
        model: denoiser 或超分 denoiser。
        diffusion: `GaussianDiffusion` 或 `GaussianDiffusionSR`。
        x_start: 普通 DDPM 中的 clean image；超分中为 GT。
        lq: 超分低分图像。普通 DDPM 时为 None。
        schedule_sampler: timestep sampler，默认均匀采样。

    Returns:
        loss: 标量 loss。
        log_dict: 用于日志的字典。
    """
    sampler = schedule_sampler or UniformSampler(diffusion)
    timesteps, weights = sampler.sample(x_start.shape[0], x_start.device)

    if lq is None:
        loss, log_dict = diffusion.training_losses(model, x_start, t=timesteps)
    else:
        loss, log_dict = diffusion.training_losses(model, x_start, lq, t=timesteps)

    # 当前 UniformSampler 权重恒为 1；保留加权形式方便后续扩展。
    weighted_loss = loss * weights.mean()
    log_dict = {k: v.detach() if torch.is_tensor(v) else v for k, v in log_dict.items()}
    return weighted_loss, log_dict
