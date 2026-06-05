import os
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from simplesr.utils.degradations import (
    circular_lowpass_kernel,
    random_mixed_kernels,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from simplesr.utils.diffjpeg import DiffJPEG
from simplesr.utils.img_process_utils import filter2D,USMSharp

from simplesr.utils.img_utils import read_img_as_rgb_float, tensor_to_img_array, img_array_to_tensor, write_rgb_float_img
# ============================================================
# 默认退化参数
#    基本参考 Real-ESRGAN x4 的退化配置
# ============================================================

DEFAULT_DEGRADATION_OPT = {
    # 超分倍率，例如 x4 表示 HR -> LR 尺寸缩小 4 倍
    "scale": 4,

    # 是否对 GT 先做 USM 锐化
    # 对自然图像可以开；对遥感图像建议先分别试 True / False
    "use_usm": True,

    # 是否启用第一阶段 JPEG 压缩退化
    "use_jpeg": False,

    # 是否启用最后阶段 JPEG 压缩退化
    "use_jpeg2": False,

    # -------------------------
    # 第一阶段退化参数
    # -------------------------
    # 第一阶段 resize 操作的采样概率，分别对应：放大 / 缩小 / 保持尺寸
    "resize_prob": [0.0, 0.0, 1.0],  # up, down, keep
    # 第一阶段 resize 的缩放因子范围
    "resize_range": [0.15, 1.5],

    # 第一阶段添加高斯噪声的概率
    "gaussian_noise_prob": 0.5,
    # 第一阶段高斯噪声强度范围，通常可理解为 sigma 范围
    "noise_range": [1, 30],
    # 第一阶段泊松噪声的缩放范围，越大噪声波动通常越明显
    "poisson_scale_range": [0.05, 3],
    # 第一阶段使用灰度噪声的概率；否则通常为彩色独立噪声
    "gray_noise_prob": 0.4,

    # 第一阶段 JPEG 压缩质量范围，值越低压缩伪影通常越重
    "jpeg_range": [30, 95],

    # -------------------------
    # 第二阶段退化参数
    # -------------------------
    # 第二阶段再次执行模糊的概率
    "second_blur_prob": 0.8,

    # 第二阶段 resize 操作的采样概率，分别对应：放大 / 缩小 / 保持尺寸
    "resize_prob2": [0.0, 0.0, 1.0],  # up, down, keep
    # 第二阶段 resize 的缩放因子范围
    "resize_range2": [0.3, 1.2],

    # 第二阶段添加高斯噪声的概率
    "gaussian_noise_prob2": 0.5,
    # 第二阶段高斯噪声强度范围
    "noise_range2": [1, 25],
    # 第二阶段泊松噪声的缩放范围
    "poisson_scale_range2": [0.05, 2.5],
    # 第二阶段使用灰度噪声的概率
    "gray_noise_prob2": 0.4,

    # 第二阶段 JPEG 压缩质量范围
    "jpeg_range2": [30, 95],

    # -------------------------
    # 第一阶段 blur kernel 参数
    # -------------------------
    # 第一阶段模糊核的最大尺寸配置；实际采样时通常会从若干奇数尺寸中选取
    "blur_kernel_size": 21,
    # 第一阶段候选模糊核类型列表：
    # isotropic / anisotropic / generalized / plateau 等
    "kernel_list": [
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    ],
    # 与 kernel_list 对应的采样概率
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    # 第一阶段直接使用 sinc 低通核替代随机模糊核的概率
    "sinc_prob": 0.1,
    # 第一阶段模糊核 sigma 范围，控制模糊强弱
    "blur_sigma": [0.2, 3],
    # generalized Gaussian kernel 的 beta 参数范围
    "betag_range": [0.5, 4],
    # plateau 型 kernel 的 beta 参数范围
    "betap_range": [1, 2],

    # -------------------------
    # 第二阶段 blur kernel 参数
    # -------------------------
    # 第二阶段模糊核的最大尺寸配置
    "blur_kernel_size2": 21,
    # 第二阶段候选模糊核类型列表
    "kernel_list2": [
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    ],
    # 与 kernel_list2 对应的采样概率
    "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    # 第二阶段直接使用 sinc 低通核替代随机模糊核的概率
    "sinc_prob2": 0.1,
    # 第二阶段模糊核 sigma 范围，通常略弱于第一阶段
    "blur_sigma2": [0.2, 1.5],
    # 第二阶段 generalized Gaussian kernel 的 beta 参数范围
    "betag_range2": [0.5, 4],
    # 第二阶段 plateau 型 kernel 的 beta 参数范围
    "betap_range2": [1, 2],

    # 最后的 sinc 低通核概率
    "final_sinc_prob": 0.8,
}


# ============================================================
#  kernel 生成函数
# ============================================================

def generate_blur_kernel(
    opt: Dict,
    stage: int = 1,
    kernel_range: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    生成第一阶段或第二阶段的 blur kernel。

    stage=1:
        使用 kernel_list, kernel_prob, blur_sigma 等参数。

    stage=2:
        使用 kernel_list2, kernel_prob2, blur_sigma2 等参数。
    """
    if kernel_range is None:
        # Real-ESRGAN 默认 kernel size 从 7 到 21，且为奇数
        kernel_range = [2 * v + 1 for v in range(3, 11)]

    kernel_size = random.choice(kernel_range)

    if stage == 1:
        sinc_prob = opt["sinc_prob"]
        kernel_list = opt["kernel_list"]
        kernel_prob = opt["kernel_prob"]
        blur_sigma = opt["blur_sigma"]
        betag_range = opt["betag_range"]
        betap_range = opt["betap_range"]
        target_kernel_size = opt["blur_kernel_size"]
    else:
        sinc_prob = opt["sinc_prob2"]
        kernel_list = opt["kernel_list2"]
        kernel_prob = opt["kernel_prob2"]
        blur_sigma = opt["blur_sigma2"]
        betag_range = opt["betag_range2"]
        betap_range = opt["betap_range2"]
        target_kernel_size = opt["blur_kernel_size2"]

    # 按概率生成 sinc 低通核
    if np.random.uniform() < sinc_prob:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)

        kernel = circular_lowpass_kernel(
            omega_c,
            kernel_size,
            pad_to=False,
        )

    # 否则生成随机混合 blur kernel
    else:
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma,
            [-math.pi, math.pi],
            betag_range,
            betap_range,
            noise_range=None,
        )

    # pad 到固定大小，方便 batch / 卷积处理
    pad_size = (target_kernel_size - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    return kernel.astype(np.float32)


def generate_final_sinc_kernel(
    opt: Dict,
    kernel_range: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    生成最后阶段的 sinc kernel。

    如果没有触发 final_sinc_prob，则返回 pulse kernel，
    也就是单位卷积核，相当于不做额外滤波。
    """
    if kernel_range is None:
        kernel_range = [2 * v + 1 for v in range(3, 11)]

    target_kernel_size = 21

    if np.random.uniform() < opt["final_sinc_prob"]:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)

        sinc_kernel = circular_lowpass_kernel(
            omega_c,
            kernel_size,
            pad_to=target_kernel_size,
        )
    else:
        # pulse kernel：中心为 1，其余为 0，相当于不滤波
        sinc_kernel = np.zeros((target_kernel_size, target_kernel_size), dtype=np.float32)
        sinc_kernel[target_kernel_size // 2, target_kernel_size // 2] = 1.0

    return sinc_kernel.astype(np.float32)


def kernel_to_tensor(kernel: np.ndarray, device: str) -> torch.Tensor:
    """
    numpy kernel -> torch kernel
    H, W -> 1, H, W
    """
    return torch.from_numpy(kernel).float().unsqueeze(0).to(device)





# ============================================================
# 4. Real-ESRGAN 风格退化器
# ============================================================

class RealESRGANStyleDegrader:
    """
    一个离线版本的 Real-ESRGAN 风格退化器。

    功能：
    1. 输入单张 RGB 图像，生成 LR 图像；
    2. 可以复用同一个对象处理大量图像；
    3. 退化流程参考 Real-ESRGAN 的 two-order degradation。
    """

    def __init__(
        self,
        opt: Optional[Dict] = None,
        device: Optional[str] = None,
    ) -> None:
        self.opt = DEFAULT_DEGRADATION_OPT.copy()
        if opt is not None:
            self.opt.update(opt)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # JPEG 模拟器
        if self.opt.get("use_jpeg", True) or self.opt.get("use_jpeg2", True):
            self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        else:
            self.jpeger = None

        # USM 锐化器
        self.usm_sharpener = USMSharp().to(self.device)

    @torch.no_grad()
    def degrade_image_array(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        对单张 RGB float32 图像进行退化。

        输入：
            img_rgb:
                H, W, 3
                RGB
                float32
                范围 [0, 1]

        输出：
            lr_rgb:
                H / scale, W / scale, 3
                RGB
                float32
                范围 [0, 1]
        """
        opt = self.opt
        scale = int(opt["scale"])

        # 转成 tensor: 1, C, H, W
        gt = img_array_to_tensor(img_rgb).to(self.device)
        ori_h, ori_w = gt.shape[2:4]

        # 为了保证最终尺寸整除 scale，这里建议输入 patch 尺寸本身可被 scale 整除
        if ori_h % scale != 0 or ori_w % scale != 0:
            raise ValueError(
                f"输入图像尺寸需要能被 scale={scale} 整除，"
                f"当前尺寸为 H={ori_h}, W={ori_w}"
            )

        # 是否先对 GT 做 USM 锐化
        if opt.get("use_usm", True):
            gt_for_degradation = self.usm_sharpener(gt)
        else:
            gt_for_degradation = gt

        # 生成三类 kernel
        kernel1 = kernel_to_tensor(generate_blur_kernel(opt, stage=1), self.device)
        kernel2 = kernel_to_tensor(generate_blur_kernel(opt, stage=2), self.device)
        sinc_kernel = kernel_to_tensor(generate_final_sinc_kernel(opt), self.device)

        # =====================================================
        # 第一阶段退化：blur -> random resize -> noise -> JPEG
        # =====================================================

        # 1. blur
        out = filter2D(gt_for_degradation, kernel1)

        # 2. random resize
        updown_type = random.choices(["up", "down", "keep"], opt["resize_prob"])[0]
        if updown_type == "up":
            resize_scale = np.random.uniform(1, opt["resize_range"][1])
        elif updown_type == "down":
            resize_scale = np.random.uniform(opt["resize_range"][0], 1)
        else:
            resize_scale = 1.0

        mode = random.choice(["area", "bilinear", "bicubic"])
        if resize_scale != 1.0:
            out = F.interpolate(out, scale_factor=resize_scale, mode=mode)

        # 3. noise
        if np.random.uniform() < opt["gaussian_noise_prob"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=opt["noise_range"],
                clip=True,
                rounds=False,
                gray_prob=opt["gray_noise_prob"],
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt["poisson_scale_range"],
                gray_prob=opt["gray_noise_prob"],
                clip=True,
                rounds=False,
            )

        # 4. JPEG
        if opt.get("use_jpeg", True):
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt["jpeg_range"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

        # =====================================================
        # 第二阶段退化：optional blur -> random resize -> noise
        # =====================================================

        # 1. optional blur
        if np.random.uniform() < opt["second_blur_prob"]:
            out = filter2D(out, kernel2)

        # 2. random resize
        updown_type = random.choices(["up", "down", "keep"], opt["resize_prob2"])[0]
        if updown_type == "up":
            resize_scale = np.random.uniform(1, opt["resize_range2"][1])
        elif updown_type == "down":
            resize_scale = np.random.uniform(opt["resize_range2"][0], 1)
        else:
            resize_scale = 1.0

        mode = random.choice(["area", "bilinear", "bicubic"])

        # 这里先缩放到接近最终 LR 尺寸的随机尺寸
        target_h = int(ori_h / scale * resize_scale)
        target_w = int(ori_w / scale * resize_scale)
        target_h = max(1, target_h)
        target_w = max(1, target_w)

        out = F.interpolate(out, size=(target_h, target_w), mode=mode)

        # 3. noise
        if np.random.uniform() < opt["gaussian_noise_prob2"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=opt["noise_range2"],
                clip=True,
                rounds=False,
                gray_prob=opt["gray_noise_prob2"],
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt["poisson_scale_range2"],
                gray_prob=opt["gray_noise_prob2"],
                clip=True,
                rounds=False,
            )

        # =====================================================
        # 最后阶段：
        #   顺序 A: resize back -> sinc -> JPEG
        #   顺序 B: JPEG -> resize back -> sinc
        # =====================================================

        final_h = ori_h // scale
        final_w = ori_w // scale
        already_at_final_size = (target_h == final_h and target_w == final_w)

        if not opt.get("use_jpeg2", True):
            if not already_at_final_size:
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(out, size=(final_h, final_w), mode=mode)
            out = filter2D(out, sinc_kernel)

        elif np.random.uniform() < 0.5:
            # 顺序 A：先 resize 到最终 LR 尺寸，再 sinc，再 JPEG
            if not already_at_final_size:
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(out, size=(final_h, final_w), mode=mode)
            out = filter2D(out, sinc_kernel)

            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

        else:
            # 顺序 B：先 JPEG，再 resize 到最终 LR 尺寸，再 sinc
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            if not already_at_final_size:
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(out, size=(final_h, final_w), mode=mode)
            out = filter2D(out, sinc_kernel)

        # 模拟 8-bit 量化
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        # tensor -> numpy RGB image
        lr_rgb = tensor_to_img_array(out)
        return lr_rgb

    # 入口函数
    def degrade_file(
        self,
        input_path: str,
        output_path: str,
    ) -> None:
        """
        对单张图像文件退化，并保存结果。
        """
        img_rgb = read_img_as_rgb_float(input_path)
        lr_rgb = self.degrade_image_array(img_rgb)
        write_rgb_float_img(lr_rgb, output_path)


class RealESRGANStyleDegraderGPU:
    """
    GPU batch 版本的 Real-ESRGAN 风格退化器。

    说明：
    1. 输入为已经切好的同尺寸 HR patch batch；
    2. 每张图独立生成 kernel1、kernel2、sinc_kernel；
    """

    def __init__(self, opt: Dict, device: str = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' 但当前环境不可用 CUDA")

        self.opt = DEFAULT_DEGRADATION_OPT.copy()
        self.opt.update(opt)
        self.device = device

        if self.opt.get("use_jpeg", True) or self.opt.get("use_jpeg2", True):
            self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        else:
            self.jpeger = None

        self.usm_sharpener = USMSharp().to(self.device)

    def _make_kernel_batch(self, batch_size: int, stage: int) -> torch.Tensor:
        kernels = [
            generate_blur_kernel(self.opt, stage=stage)
            for _ in range(batch_size)
        ]
        kernel_np = np.stack(kernels, axis=0).astype(np.float32)
        return torch.from_numpy(kernel_np).to(self.device)

    def _make_sinc_kernel_batch(self, batch_size: int) -> torch.Tensor:
        kernels = [
            generate_final_sinc_kernel(self.opt)
            for _ in range(batch_size)
        ]
        kernel_np = np.stack(kernels, axis=0).astype(np.float32)
        return torch.from_numpy(kernel_np).to(self.device)

    @torch.no_grad()
    def degrade_batch_tensor(self, gt: torch.Tensor) -> torch.Tensor:
        """
        对一批同尺寸 HR patch 做 GPU 退化。

        输入：
            gt: [B, C, H, W], RGB, float32, [0, 1]

        输出：
            lq: [B, C, H/scale, W/scale], RGB, float32, [0, 1]
        """
        if gt.ndim != 4:
            raise ValueError(f"gt 必须是 4 维张量 [B, C, H, W]，当前 shape={tuple(gt.shape)}")

        gt = gt.to(self.device, non_blocking=True).float()
        b, _, ori_h, ori_w = gt.shape
        scale = int(self.opt["scale"])

        if ori_h % scale != 0 or ori_w % scale != 0:
            raise ValueError(
                f"输入图像尺寸需要能被 scale={scale} 整除，当前尺寸为 H={ori_h}, W={ori_w}"
            )

        if self.opt.get("use_usm", True):
            gt_for_degradation = self.usm_sharpener(gt)
        else:
            gt_for_degradation = gt

        kernel1_batch = self._make_kernel_batch(b, stage=1)
        kernel2_batch = self._make_kernel_batch(b, stage=2)
        sinc_kernel_batch = self._make_sinc_kernel_batch(b)

        # 第一阶段先统一做批量 blur
        out_batch = filter2D(gt_for_degradation, kernel1_batch)
        results = []

        for idx in range(b):
            out = out_batch[idx:idx + 1]
            kernel2 = kernel2_batch[idx:idx + 1]
            sinc_kernel = sinc_kernel_batch[idx:idx + 1]

            # 第一阶段 random resize
            updown_type = random.choices(["up", "down", "keep"], self.opt["resize_prob"])[0]
            if updown_type == "up":
                resize_scale = np.random.uniform(1, self.opt["resize_range"][1])
            elif updown_type == "down":
                resize_scale = np.random.uniform(self.opt["resize_range"][0], 1)
            else:
                resize_scale = 1.0

            mode = random.choice(["area", "bilinear", "bicubic"])
            if resize_scale != 1.0:
                out = F.interpolate(out, scale_factor=resize_scale, mode=mode)

            # 第一阶段 noise
            if np.random.uniform() < self.opt["gaussian_noise_prob"]:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range"],
                    clip=True,
                    rounds=False,
                    gray_prob=self.opt["gray_noise_prob"],
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range"],
                    gray_prob=self.opt["gray_noise_prob"],
                    clip=True,
                    rounds=False,
                )

            # 第一阶段 JPEG
            if self.opt.get("use_jpeg", True):
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)

            # 第二阶段 optional blur
            if np.random.uniform() < self.opt["second_blur_prob"]:
                out = filter2D(out, kernel2)

            # 第二阶段 random resize
            updown_type = random.choices(["up", "down", "keep"], self.opt["resize_prob2"])[0]
            if updown_type == "up":
                resize_scale = np.random.uniform(1, self.opt["resize_range2"][1])
            elif updown_type == "down":
                resize_scale = np.random.uniform(self.opt["resize_range2"][0], 1)
            else:
                resize_scale = 1.0

            mode = random.choice(["area", "bilinear", "bicubic"])
            target_h = max(1, int(ori_h / scale * resize_scale))
            target_w = max(1, int(ori_w / scale * resize_scale))
            out = F.interpolate(out, size=(target_h, target_w), mode=mode)

            # 第二阶段 noise
            if np.random.uniform() < self.opt["gaussian_noise_prob2"]:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range2"],
                    clip=True,
                    rounds=False,
                    gray_prob=self.opt["gray_noise_prob2"],
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range2"],
                    gray_prob=self.opt["gray_noise_prob2"],
                    clip=True,
                    rounds=False,
                )

            final_h = ori_h // scale
            final_w = ori_w // scale
            already_at_final_size = (target_h == final_h and target_w == final_w)

            if not self.opt.get("use_jpeg2", True):
                if not already_at_final_size:
                    mode = random.choice(["area", "bilinear", "bicubic"])
                    out = F.interpolate(out, size=(final_h, final_w), mode=mode)
                out = filter2D(out, sinc_kernel)
            elif np.random.uniform() < 0.5:
                if not already_at_final_size:
                    mode = random.choice(["area", "bilinear", "bicubic"])
                    out = F.interpolate(out, size=(final_h, final_w), mode=mode)
                out = filter2D(out, sinc_kernel)

                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)

                if not already_at_final_size:
                    mode = random.choice(["area", "bilinear", "bicubic"])
                    out = F.interpolate(out, size=(final_h, final_w), mode=mode)
                out = filter2D(out, sinc_kernel)

            out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
            results.append(out)

        return torch.cat(results, dim=0)


# ============================================================
# 5. 单张图像退化函数
# ============================================================

def degrade_single_image(
    input_path: str,
    output_path: str,
    scale: int = 4,
    opt: Optional[Dict] = None,
    device: Optional[str] = None,
) -> None:
    """
    输入单张 HR 图像，生成对应 LR 图像。

    示例：
        degrade_single_image(
            input_path="HR/a.png",
            output_path="LR/a.png",
            scale=4,
        )
    """
    final_opt = {"scale": scale}
    if opt is not None:
        final_opt.update(opt)

    degrader = RealESRGANStyleDegrader(opt=final_opt, device=device)
    degrader.degrade_file(input_path, output_path)


# ============================================================
# 6. 文件夹退化：单进程版本
# ============================================================

def collect_image_paths(
    input_dir: str,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    recursive: bool = True,
) -> Sequence[Path]:
    """
    收集文件夹下所有图像路径。
    """
    input_dir = Path(input_dir)

    if recursive:
        paths = [
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        ]
    else:
        paths = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]

    return sorted(paths)


def degrade_folder(
    input_dir: str,
    output_dir: str,
    scale: int = 4,
    opt: Optional[Dict] = None,
    device: Optional[str] = None,
    recursive: bool = True,
) -> None:
    """
    输入 HR 图像文件夹，生成结构一致的 LR 图像文件夹。

    示例：
        degrade_folder(
            input_dir="datasets/HR",
            output_dir="datasets/LR_x4",
            scale=4,
            recursive=True,
        )
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_opt = {"scale": scale}
    if opt is not None:
        final_opt.update(opt)

    degrader = RealESRGANStyleDegrader(opt=final_opt, device=device)
    image_paths = collect_image_paths(str(input_dir), recursive=recursive)

    print(f"发现 {len(image_paths)} 张图像，开始退化...")

    for input_path in tqdm(image_paths, desc="Degrading images"):
        rel_path = input_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        degrader.degrade_file(str(input_path), str(output_path))

    print(f"完成，LR 图像保存到: {output_dir}")


def degrade_folder_gpu(
    input_dir: str,
    output_dir: str,
    scale: int = 4,
    opt: Optional[Dict] = None,
    batch_size: int = 16,
    recursive: bool = True,
    device: str = "cuda",
) -> None:
    """
    GPU batch 版文件夹退化。

    推荐：
        - 已切好的同尺寸 HR patch：优先使用这个函数；
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' 但当前环境不可用 CUDA")

    torch.backends.cudnn.benchmark = True

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_opt = {"scale": scale}
    if opt is not None:
        final_opt.update(opt)

    degrader = RealESRGANStyleDegraderGPU(opt=final_opt, device=device)
    image_paths = collect_image_paths(str(input_dir), recursive=recursive)

    print(f"发现 {len(image_paths)} 张图像，batch_size={batch_size}，使用 {device} 开始退化...")

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Degrading images (GPU batch)"):
        batch_paths = image_paths[start:start + batch_size]
        imgs_rgb = [read_img_as_rgb_float(str(path)) for path in batch_paths]
        gt_batch = img_array_to_tensor(imgs_rgb).to(device, non_blocking=True)
        lq_batch = degrader.degrade_batch_tensor(gt_batch)
        lq_imgs = tensor_to_img_array(lq_batch)

        for input_path, lq_img in zip(batch_paths, lq_imgs):
            rel_path = input_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            write_rgb_float_img(lq_img, str(output_path))

    print(f"完成，LR 图像保存到: {output_dir}")


# ============================================================
# 7. 使用示例
# ============================================================

if __name__ == "__main__":
    # 示例 1：单张图像退化
    # degrade_single_image(
    #     input_path=r"D:\datasets\RS_HR\a.png",
    #     output_path=r"D:\datasets\RS_LR_x4\a.png",
    #     scale=4,
    #     device="cuda",
    # )

    # 示例 2：文件夹退化，单进程，可用 GPU
    degrade_folder(
        input_dir=r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train\images",
        output_dir=r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train_lrx4\images",
        scale=4,
        device="cuda",
        recursive=True,
    )

    degrade_folder(
        input_dir=r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\val\images",
        output_dir=r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\val_lrx4\images",
        scale=4,
        device="cuda",
        recursive=True,
    )


    # 示例 3：用 GPU batch 生成 LR 图像
    #
    # degrade_folder_gpu(
    #     input_dir=r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train\images",
    #     output_dir=r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train_lrx4\images",
    #     scale=4,
    #     batch_size=16,
    #     recursive=True,
    #     device="cuda",
    # )
    pass
