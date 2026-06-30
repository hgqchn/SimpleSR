"""DDPM baseline loss 工具。

当前精简版只使用 MSE 训练目标。该文件保留为后续扩展 VLB、KL 或感知损失入口。
"""

import torch

from .nn import mean_flat


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """返回每个样本的 MSE，shape 为 ``(B,)``。"""
    return mean_flat((pred - target) ** 2)
