import sys
import os

import torch.nn as nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(1,channel // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, mid_channels, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    # B,C,H,W -> B,C,H,W
    def forward(self, x):
        tmp = x
        #b, c, _, _ = x.size()
        # b,c,1,1
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y
# ResSe
class ResSELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(ResSELayer, self).__init__()
        self.se=SELayer(channel, reduction)

    # B,C,H,W -> B,C,H,W
    def forward(self, x):
        return self.se(x)+x

# CBAM
class CBAM_CA(nn.Module):
    """CBAM 通道注意力：AvgPool + MaxPool -> 共享MLP -> Sigmoid"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp(self.avg_pool(x))
        m = self.mlp(self.max_pool(x))
        w = self.act(a + m)
        return x * w

class CBAM_SA(nn.Module):
    """CBAM 空间注意力：沿通道做 Avg/Max，再拼接 -> 7x7 Conv -> Sigmoid"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError(...)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #B,1,H,W
        avg = torch.mean(x, dim=1, keepdim=True)
        #B,1,H,W
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.act(self.conv(torch.cat([avg, max_map], dim=1)))
        return x * attn

class CBAM(nn.Module):
    """CBAM 主模块：先通道注意力，再空间注意力"""
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = CBAM_CA(channels, reduction)
        self.sa = CBAM_SA(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x

# CoordAttention
class CoordAtt(nn.Module):
    """
    Coordinate Attention (CVPR'21)
    输入:  x: (B, C, H, W)
    输出:  y: (B, C, H, W)
    关键:  沿 W/H 分别聚合，保留 H/W 的位置信息；轻量且可导出
    """
    def __init__(self, channels: int, reduction: int = 32, use_hs: bool = True):
        super().__init__()
        # 论文建议: bottleneck 至少 8 个通道
        mid = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.act   = nn.Hardswish() if use_hs else nn.ReLU(inplace=True)

        # 两个 1x1 将 mid -> C，分别生成 H/W 分支权重
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # 沿宽度求均值 -> 保留 H 维: (B,C,H,1)
        x_h = x.mean(dim=3, keepdim=True)
        # 沿高度求均值 -> 保留 W 维: (B,C,1,W) -> 置换成 (B,C,W,1) 便于沿“空间长轴”拼接
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        # 串接后做共享变换: (B,C,H+W,1) -> (B,mid,H+W,1)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        # 切回两条坐标分支
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # (B,mid,1,W)

        a_h = self.sigmoid(self.conv_h(y_h))  # (B,C,H,1)
        a_w = self.sigmoid(self.conv_w(y_w))  # (B,C,1,W)

        return x * a_h * a_w


class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module (ICML 2021)
    - 每层不引入可学习参数（仅使用统计量）
    - lambda_param(λ): 能量函数中的系数，官方示例常用 0.1
    """
    def __init__(self, lambda_param: float = 0.1):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        n = h * w - 1 if h * w > 1 else 1


        # 通道内均值 -> (B,C,1,1)，广播求 (x - mean)^2
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        d = (x - x_mean).pow(2)   # (B,C,H,W)

        # 通道方差估计 v -> (B,C,1,1)
        v = d.sum(dim=[2, 3], keepdim=True) / n

        # 闭式解的“逆能量” E_inv（README 伪代码）
        e_inv = d / (4 * (v + self.lambda_param)) + 0.5  # (B,C,H,W)

        # 3D 注意力权重
        attn = torch.sigmoid(e_inv)
        return x * attn



if __name__ == '__main__':
    pass
