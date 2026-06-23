from copy import deepcopy

from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_psnr_pt, calculate_ssim, calculate_ssim_pt

__all__ = ['calculate_metric', 'calculate_psnr', 'calculate_psnr_pt', 'calculate_ssim', 'calculate_ssim_pt',
           'calculate_niqe']


METRIC_FUNCS = {
    'calculate_psnr': calculate_psnr,
    'calculate_psnr_pt': calculate_psnr_pt,
    'calculate_ssim': calculate_ssim,
    'calculate_ssim_pt': calculate_ssim_pt,
    'calculate_niqe': calculate_niqe,
    'psnr': calculate_psnr,
    'PSNR': calculate_psnr,
    'ssim': calculate_ssim,
    'SSIM': calculate_ssim,
    'niqe': calculate_niqe,
    'NIQE': calculate_niqe,
}


def calculate_metric(data, opt):
    """根据配置计算指标。

    参数:
        data (dict): 指标输入数据。不同指标需要的字段不同：
            - 全参考指标（PSNR / SSIM）需要 ``img`` 和 ``img2``：
              ``img`` 通常是模型输出图像，``img2`` 是 GT 图像。
            - 无参考指标（NIQE）只需要 ``img``。
            图像通常为 numpy.ndarray，支持 HWC 或 CHW 顺序；当前项目验证流程中
            常见输入是 RGB float 图像，范围为 [0, 1]。

        opt (dict): 指标配置，必须包含 ``type`` 字段，用来指定指标函数。
            其它字段会作为关键字参数传给具体指标函数，例如：
            - ``crop_border``: 计算前裁剪图像边缘像素数。
            - ``input_order``: 输入图像顺序，常用 ``'HWC'`` 或 ``'CHW'``。
            - ``test_y_channel``: PSNR / SSIM 是否只在 Y 通道上计算。

            示例:
                ``{'type': 'calculate_psnr', 'crop_border': 4, 'test_y_channel': False}``
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    if metric_type not in METRIC_FUNCS:
        raise KeyError(f'Unsupported metric type: {metric_type}. Supported metrics: {tuple(METRIC_FUNCS)}')
    return METRIC_FUNCS[metric_type](**data, **opt)
