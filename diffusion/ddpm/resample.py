"""训练 timestep 采样器。

为了保持 baseline 简洁，这里只保留均匀 timestep 采样。
Improved DDPM 中的 loss-second-moment 重采样已删除；后续确实需要时再添加。
"""

import torch


class UniformSampler:
    """均匀采样扩散 timestep。"""

    def __init__(self, diffusion):
        self.diffusion = diffusion

    def sample(self, batch_size: int, device: torch.device):
        timesteps = torch.randint(
            low=0,
            high=self.diffusion.num_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        weights = torch.ones(batch_size, device=device)
        return timesteps, weights


def create_named_schedule_sampler(name: str, diffusion):
    """根据名称创建 timestep sampler。"""
    if name != "uniform":
        raise ValueError(f"Only uniform schedule sampler is kept, got: {name}")
    return UniformSampler(diffusion)
