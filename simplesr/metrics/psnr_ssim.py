import cv2
import numpy as np
import torch
import torch.nn.functional as F

from simplesr.metrics.metric_util import img_to_255, reorder_image, to_y_channel
from simplesr.utils.color_utils import rgb2ycbcr_pt


def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """计算 PSNR（Peak Signal-to-Noise Ratio，峰值信噪比）。

    当前项目中的验证图像通常是 RGB float，范围为 [0, 1]；
    若检测到该范围，会自动转换到 [0, 255] 后计算。

    参数:
        img (ndarray): 第一张图像，范围可为 [0, 1] 或 [0, 255]。
        img2 (ndarray): 第二张图像，范围可为 [0, 1] 或 [0, 255]。
        crop_border (int): 每条边裁剪的像素数，裁剪区域不参与计算。
        input_order (str): 输入顺序，支持 'HWC' 或 'CHW'。默认值：'HWC'。
        test_y_channel (bool): 是否只在 YCbCr 的 Y 通道上计算。默认值：False。

    返回:
        float: PSNR 结果。
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img_to_255(img)
    img2 = img_to_255(img2)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)


def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """计算 PyTorch 版本 PSNR。

    参数:
        img (Tensor): 范围为 [0, 1] 的图像，形状为 (N, C, H, W)。
        img2 (Tensor): 范围为 [0, 1] 的图像，形状为 (N, C, H, W)。
        crop_border (int): 每条边裁剪的像素数。
        test_y_channel (bool): 是否只在 Y 通道上计算。默认值：False。

    返回:
        Tensor: 每张图像的 PSNR。
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """计算 SSIM（Structural Similarity，结构相似性）。

    当前实现与经典 MATLAB SSIM 计算流程保持一致。

    当前项目中的验证图像通常是 RGB float，范围为 [0, 1]；
    若检测到该范围，会自动转换到 [0, 255] 后计算。

    三通道图像会逐通道计算 SSIM 后取平均。

    参数:
        img (ndarray): 第一张图像，范围可为 [0, 1] 或 [0, 255]。
        img2 (ndarray): 第二张图像，范围可为 [0, 1] 或 [0, 255]。
        crop_border (int): 每条边裁剪的像素数。
        input_order (str): 输入顺序，支持 'HWC' 或 'CHW'。默认值：'HWC'。
        test_y_channel (bool): 是否只在 Y 通道上计算。默认值：False。

    返回:
        float: SSIM 结果。
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img_to_255(img)
    img2 = img_to_255(img2)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


def calculate_ssim_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """计算 PyTorch 版本 SSIM。

    参数:
        img (Tensor): 范围为 [0, 1] 的图像，形状为 (N, C, H, W)。
        img2 (Tensor): 范围为 [0, 1] 的图像，形状为 (N, C, H, W)。
        crop_border (int): 每条边裁剪的像素数。
        test_y_channel (bool): 是否只在 Y 通道上计算。默认值：False。

    返回:
        Tensor: 每张图像的 SSIM。
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)
    return ssim


def _ssim(img, img2):
    """计算单通道图像的 SSIM。

    参数:
        img (ndarray): 范围为 [0, 255] 的单通道图像。
        img2 (ndarray): 范围为 [0, 255] 的单通道图像。

    返回:
        float: SSIM 结果。
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def _ssim_pth(img, img2):
    """计算 PyTorch 版本单批次 SSIM。

    参数:
        img (Tensor): 范围为 [0, 255] 的图像，形状为 (N, C, H, W)。
        img2 (Tensor): 范围为 [0, 255] 的图像，形状为 (N, C, H, W)。

    返回:
        Tensor: 每张图像的 SSIM。
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])
