import numpy as np

from simplesr.utils.color_utils import rgb2ycbcr


def reorder_image(img, input_order='HWC'):
    """将图像调整为 HWC 顺序。

    如果输入是 (H, W)，返回 (H, W, 1)；
    如果输入是 (C, H, W)，返回 (H, W, C)；
    如果输入已经是 (H, W, C)，则原样返回。

    参数:
        img (ndarray): 输入图像。
        input_order (str): 输入图像顺序，支持 'HWC' 和 'CHW'。
            当输入是二维图像时，该参数不生效。默认值：'HWC'。

    返回:
        ndarray: HWC 顺序的图像。
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """转换到 YCbCr 的 Y 通道。

    参数:
        img (ndarray): 范围为 [0, 255] 的 RGB 图像。

    返回:
        ndarray: 范围为 [0, 255] 的 Y 通道图像，float 类型，不取整。
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = rgb2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def img_to_255(img):
    """将当前项目常用的 [0, 1] RGB float 图像兼容转换到 [0, 255]。"""
    img = img.astype(np.float32, copy=False)
    if img.size > 0 and img.max() <= 1.0:
        img = img * 255.0
    return img
