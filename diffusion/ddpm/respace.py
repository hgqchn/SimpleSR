"""少步采样工具。

该文件保留 guided-diffusion / IDDPM 风格的 timestep respacing 思路：
从原始 T 步扩散中选出较少的 timestep，并构造等效 beta schedule。
"""

from __future__ import annotations

import numpy as np
import torch

from diffusion.common_utils import get_named_beta_schedule
from .gaussian_diffusion import GaussianDiffusion, GaussianDiffusionSR


def space_timesteps(num_timesteps: int, section_counts) -> set[int]:
    """从原始扩散步中选择少量 timestep。

    Args:
        num_timesteps: 原始扩散步数。
        section_counts: 可以是整数、逗号分隔字符串、列表，或 ``ddimN``。
    """
    if isinstance(section_counts, int):
        section_counts = [section_counts]
    elif isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired = int(section_counts[len("ddim") :])
            for stride in range(1, num_timesteps):
                steps = set(range(0, num_timesteps, stride))
                if len(steps) == desired:
                    return steps
            raise ValueError(f"Cannot create exactly {desired} steps from {num_timesteps}")
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < count:
            raise ValueError(f"Cannot divide section of {size} steps into {count}")
        stride = 1 if count <= 1 else (size - 1) / (count - 1)
        cur = 0.0
        for _ in range(count):
            all_steps.append(start_idx + round(cur))
            cur += stride
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """使用 timestep 子集的 DDPM。"""

    def __init__(
        self,
        use_timesteps=None,
        timestep_respacing="",
        betas=None,
        diffusion_steps: int = 1000,
        noise_schedule: str = "linear",
        **kwargs,
    ):
        if betas is None:
            betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_timesteps is None:
            use_timesteps = space_timesteps(len(betas), timestep_respacing or diffusion_steps)
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(betas)

        base = GaussianDiffusion(betas=betas, **kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        super().__init__(betas=np.array(new_betas, dtype=np.float64), **kwargs)

    def call_model(self, model, x_t, t, **model_kwargs):
        mapped_t = self.map_timesteps(t)
        return model(x_t, mapped_t, **model_kwargs)

    def map_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        mapping = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        return mapping[t]


class SpacedDiffusionSR(GaussianDiffusionSR):
    """使用 timestep 子集的超分 DDPM。"""

    def __init__(
        self,
        use_timesteps=None,
        timestep_respacing="",
        betas=None,
        diffusion_steps: int = 1000,
        noise_schedule: str = "linear",
        **kwargs,
    ):
        if betas is None:
            betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_timesteps is None:
            use_timesteps = space_timesteps(len(betas), timestep_respacing or diffusion_steps)
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(betas)

        base = GaussianDiffusionSR(betas=betas, **kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        super().__init__(betas=np.array(new_betas, dtype=np.float64), **kwargs)

    def call_model(self, model, x_t, t, **model_kwargs):
        mapped_t = self.map_timesteps(t)
        return model(x_t, mapped_t, **model_kwargs)

    def map_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        mapping = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        return mapping[t]
