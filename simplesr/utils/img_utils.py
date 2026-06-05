from typing import Dict, List, Tuple, Union, Literal
import cv2
import math
import numpy as np
import os
import torch
import random

def img_from_bytes(content):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.

    Returns:
        img (np.ndarray):
            float32 图像数组，数值范围 [0, 1]。
            1. 彩色图: RGB，形状 (H, W, 3)。
            2. 灰度图: 单通道，形状 (H, W, 1)。
            3. 四通道图: 丢弃 alpha 后转 RGB，形状 (H, W, 3)。
    """
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"暂不支持该图像维度: {img.shape}")

    img = img_uint8_to_float32(img)

    return img

def read_img_as_rgb_float(image_path: str) -> np.ndarray:
    """
    功能说明:
        读取图像文件并标准化为 float32 图像。

    输入参数:
        image_path (str):
            图像文件路径。

    输出参数:
        img (np.ndarray):
            float32 图像数组，数值范围 [0, 1]。
            1. 彩色图: RGB，形状 (H, W, 3)。
            2. 灰度图: 单通道，形状 (H, W, 1)。
            3. 四通道图: 丢弃 alpha 后转 RGB，形状 (H, W, 3)。
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"暂不支持该图像维度: {img.shape}, path={image_path}")

    if img.dtype == np.uint8:
        img = img_uint8_to_float32(img)
    else:
        info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
        if info is not None:
            img = img.astype(np.float32) / float(info.max)
        else:
            raise TypeError(
                f"Unsupported image dtype: {img.dtype}. "
                "Expected uint8 or integer image. "
                "If your image is float, please normalize it explicitly before degradation."
            )

    return img


def write_rgb_float_img(img_rgb: np.ndarray, save_path: str) -> None:
    """
    功能说明:
        将 RGB float 图像保存为 8-bit 图像文件。

    输入参数:
        img_rgb (np.ndarray):
            RGB 图像数组，形状 (H, W, 3)，建议范围 [0, 1]。
        save_path (str):
            输出图像路径。

    输出参数:
        无。函数将图像写入磁盘。
    """
    img_uint8 = img_float32_to_uint8(img_rgb)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    _write_img(img_bgr, str(save_path))


def _write_img(
    img: np.ndarray,
    file_path: str,
    params: Union[None, List[int]] = None,
    auto_mkdir: bool = True
) -> None:
    """
    功能说明:
        将图像数组写入文件。

    输入参数:
        img (np.ndarray):
            待写入图像数组，常见形状为 (H, W)、(H, W, 1) 或 (H, W, 3)。
            若为彩色图，默认按 OpenCV 约定使用 BGR 通道顺序。
        file_path (str):
            输出文件路径。
        params (Union[None, List[int]]):
            OpenCV `imwrite` 编码参数。
        auto_mkdir (bool):
            父目录不存在时是否自动创建。

    输出参数:
        无。函数将图像写入磁盘。
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError("Failed in writing images.")


def img_uint8_to_float32(img: np.ndarray) -> np.ndarray:
    """
    功能说明:
        将 uint8 图像转换为 float32 图像。

    输入参数:
        img (np.ndarray):
            uint8 图像数组，形状可为 (H, W)、(H, W, 1) 或 (H, W, C)，范围 [0, 255]。

    输出参数:
        out (np.ndarray):
            float32 图像数组，形状与输入一致，范围 [0, 1]。
    """
    return (img / 255.0).astype(np.float32)


def img_float32_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    功能说明:
        将 float 图像转换为 uint8 图像。

    输入参数:
        img (np.ndarray):
            float 图像数组，形状可为 (H, W)、(H, W, 1) 或 (H, W, C)。
            函数内部会先 clip 到 [0, 1]。

    输出参数:
        out (np.ndarray):
            uint8 图像数组，形状与输入一致，范围 [0, 255]。
    """
    return (img.clip(0, 1) * 255.0).round().astype(np.uint8)

def img_array_to_tensor(images_rgb: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
    """
    功能说明:
        将单张或多张 RGB numpy 图像转换为 torch batch 张量。

    输入参数:
        images_rgb (Union[np.ndarray, List[np.ndarray]]):
            1. 单张图像: 形状 (H, W, C)。
            2. 多张图像: 每个元素形状为 (H, W, C)，且所有图像 shape 必须一致。

    输出参数:
        batch (torch.Tensor):
            形状 (B, C, H, W)，dtype=float32。
            当输入为单张图像时，B=1。
    """
    if isinstance(images_rgb, np.ndarray):
        images_rgb = [images_rgb]

    if not images_rgb:
        raise ValueError("images_rgb 不能为空")

    shapes = [img.shape for img in images_rgb]
    first_shape = shapes[0]
    if any(shape != first_shape for shape in shapes):
        raise ValueError(f"一个 batch 内所有图像 shape 必须一致，当前 shapes={shapes}")

    tensors = [torch.from_numpy(img).permute(2, 0, 1).float() for img in images_rgb]
    return torch.stack(tensors, dim=0)


def tensor_to_img_array(tensor: torch.Tensor) -> Union[np.ndarray, List[np.ndarray]]:
    """
    功能说明:
        将 torch 张量转换为 RGB numpy 图像（单张或列表）。

    输入参数:
        tensor (torch.Tensor):
            1. 4 维: 形状 (B, C, H, W)。
            2. 3 维: 形状 (C, H, W)。
            数值通常在 [0, 1]，函数内部会 clamp 到 [0, 1]。

    输出参数:
        out (Union[np.ndarray, List[np.ndarray]]):
            1. 当输入为 4 维且 B=1: 返回单张图像，形状 (H, W, C)。
            2. 当输入为 4 维且 B>1: 返回图像列表，每个元素形状 (H, W, C)。
            3. 当输入为 3 维: 返回单张图像，形状 (H, W, C)。
    """
    tensor = tensor.detach().float().cpu().clamp(0, 1)

    if tensor.ndim == 4:
        if tensor.shape[0] == 1:
            return tensor[0].permute(1, 2, 0).numpy()
        return [img.permute(1, 2, 0).numpy() for img in tensor]

    if tensor.ndim == 3:
        return tensor.permute(1, 2, 0).numpy()

    raise ValueError(f"tensor 必须是 3D 或 4D，当前 shape={tuple(tensor.shape)}")


def crop_img_by_border(
    img: np.ndarray,
    crop_border: int
) -> np.ndarray:
    """
    功能说明:
        按给定像素数裁剪图像四周边缘。

    输入参数:
        img np.ndarray:
            单张图像
            常见形状为 (H, W)、(H, W, 1) 或 (H, W, C)。
        crop_border (int):
            每条边需要裁剪的像素数。

    输出参数:
        out np.ndarray:
            裁剪后的图像，类型结构与输入一致。
    """
    if crop_border == 0:
        return img
    return img[crop_border:-crop_border, crop_border:-crop_border, ...]

def crop_img_by_shape(
    img: np.ndarray,
    crop_shape: tuple[int,int],
    mode: Literal["default", "center"] = "default",
) -> np.ndarray:
    """
    功能说明:
        裁剪图像，使输出宽高均可被 `scale` 整除。

    输入参数:
        img (np.ndarray):
            输入图像，形状为 (H, W)、(H, W, 1) 或 (H, W, C)。
        crop_shape (int,int):
            裁剪后的H，W
        mode (Literal["default", "center"]):
            裁剪起点模式。
            1. default: 从左上角开始裁剪。
            2. center: 从中心开始裁剪。

    输出参数:
        img_crop (np.ndarray):
            裁剪后的图像，形状为 (new_h, new_w, ...)。
    """
    new_h,new_w = crop_shape

    assert img.ndim==2 or img.ndim==3
    height, width = img.shape[0], img.shape[1]

    if new_h>height or new_w>width:
        raise ValueError(" ")

    h_start, w_start = get_crop_start(height, width, new_h, new_w, mode)

    h_end = h_start + new_h
    w_end = w_start + new_w

    img_crop = img[h_start:h_end, w_start:w_end, ...]
    return img_crop

def crop_img_by_scale(
    img: np.ndarray,
    scale: int,
    mode: Literal["default", "center"] = "default",
) -> np.ndarray:
    """
    功能说明:
        裁剪图像，使输出宽高均可被 `scale` 整除。

    输入参数:
        img (np.ndarray):
            输入图像，形状为 (H, W)、(H, W, 1) 或 (H, W, C)。
        scale (int):
            缩放倍率，必须大于 0。
        mode (Literal["default", "center"]):
            裁剪起点模式。
            1. default: 从左上角开始裁剪。
            2. center: 从中心开始裁剪。

    输出参数:
        img_crop (np.ndarray):
            裁剪后的图像，形状为 (new_h, new_w, ...)。
            其中 new_h 与 new_w 均可被 `scale` 整除。
    """
    if scale <= 0:
        raise ValueError(f"scale 必须大于 0，但得到 {scale}")

    assert img.ndim==2 or img.ndim==3
    height, width = img.shape[0], img.shape[1]
    new_h = height // scale * scale
    new_w = width // scale * scale

    h_start, w_start = get_crop_start(height, width, new_h, new_w, mode)

    h_end = h_start + new_h
    w_end = w_start + new_w

    img_crop = img[h_start:h_end, w_start:w_end, ...]
    return img_crop


def get_crop_start(
    height: int,
    width: int,
    crop_h: int,
    crop_w: int,
    mode: Literal["default", "center"] = "default",
) -> tuple[int, int]:
    """
    功能说明:
        根据裁剪模式计算裁剪起点坐标。

    输入参数:
        height (int):
            原图高度 H。
        width (int):
            原图宽度 W。
        crop_h (int):
            裁剪高度。
        crop_w (int):
            裁剪宽度。
        mode (Literal["default", "center"]):
            裁剪起点模式。

    输出参数:
        h_start (int):
            裁剪起点的行坐标。
        w_start (int):
            裁剪起点的列坐标。
    """
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"裁剪尺寸必须大于 0，但得到 crop_h={crop_h}, crop_w={crop_w}")

    if height < crop_h or width < crop_w:
        raise ValueError(
            f"图像尺寸小于裁剪尺寸: image=({height}, {width}), crop=({crop_h}, {crop_w})"
        )

    if mode == "center":
        h_start = (height - crop_h) // 2
        w_start = (width - crop_w) // 2
        return h_start, w_start
    return 0, 0



def paired_random_crop_numpy(img_gt, img_lq, gt_patch_size, scale, gt_path=None):
    """对单张或多张 Numpy 图像执行成对随机裁剪。

    参数:
        img_gt (np.ndarray): GT 图像
        img_lq (np.ndarray): LQ 图像
        gt_patch_size (int): GT 裁剪块大小。
        scale (int): GT 与 LQ 的放大倍率。
        gt_path (str | None): GT 路径，仅用于异常信息。

    返回:
        tuple[np.ndarray, np.ndarray]:
            裁剪后的 GT/LQ 图像。返回单张图像；
    """
    h_lq, w_lq = img_lq.shape[0:2]
    h_gt, w_gt = img_gt.shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq})."
        )
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        msg = (
            f"LQ ({h_lq}, {w_lq}) is smaller than patch size "
            f"({lq_patch_size}, {lq_patch_size})."
        )
        if gt_path is not None:
            msg += f" Please remove {gt_path}."
        raise ValueError(msg)

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    img_lq = img_lq[top:top + lq_patch_size, left:left + lq_patch_size, ...]
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gt = img_gt[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]

    return img_gt, img_lq


def paired_random_crop_tensor(img_gts, img_lqs, gt_patch_size, scale):
    """对 B C H W 形状的张量执行成对随机裁剪。

    参数:
        img_gts (torch.Tensor): GT 张量，形状为 (B, C, H, W) 或 (C,H,W)。
        img_lqs (torch.Tensor): LQ 张量，形状为 (B, C, H, W) 或 (C,H,W)。
        gt_patch_size (int): GT 裁剪块大小。
        scale (int): GT 与 LQ 的放大倍率。

    返回:
        tuple[torch.Tensor, torch.Tensor]: 裁剪后的 GT/LQ 张量，形状与输入同 (B, C, H, W)。
    """
    if not (torch.is_tensor(img_gts) and torch.is_tensor(img_lqs)):
        raise TypeError("paired_random_crop_tensor 仅支持 torch.Tensor 输入")

    h_lq, w_lq = img_lqs.shape[-2:]
    h_gt, w_gt = img_gts.shape[-2:]

    lq_patch_size = gt_patch_size // scale
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq})."
        )

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    top_gt, left_gt = int(top * scale), int(left * scale)
    img_lqs = img_lqs[:, :, top:top + lq_patch_size, left:left + lq_patch_size]
    img_gts = img_gts[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size]
    return img_gts, img_lqs


def img_augment(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)

    '''

    if mode == 0:
        out = img
    elif mode == 1:
        out = np.rot90(img, k=1)
    elif mode == 2:
        out = np.flipud(np.rot90(img, k=1))
    elif mode == 3:
        out = np.rot90(img, k=2)
    elif mode == 4:
        out = np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        out = np.rot90(img, k=3)
    elif mode == 6:
        out = np.flipud(np.rot90(img, k=3))
    elif mode == 7:
        out = np.flipud(img)
    else:
        raise ValueError(f"mode must be in [0, 7], but got {mode}")


    img = np.ascontiguousarray(out)

    return img

def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def transform_points_by_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """使用 2x3 仿射矩阵变换二维点坐标。

    参数:
        points (np.ndarray):
            点坐标数组，最后一维必须表示 ``(x, y)``。
            常见形状:
            - ``(N, 2)``: N 个点；
            - ``(4, 2)``: DOTA 单个目标的四点框；
            - ``(N, 4, 2)``: N 个 DOTA 目标的四点框。
        matrix (np.ndarray):
            形状为 ``(2, 3)`` 的仿射矩阵，表示:
            ``x' = a*x + b*y + c``，``y' = d*x + e*y + f``。

    返回:
        np.ndarray:
            变换后的点坐标，形状与输入 ``points`` 一致。
    """
    # 转成 float32 的 numpy 数组，避免整数坐标在矩阵运算中丢失小数。
    points = np.asarray(points, dtype=np.float32)
    # 保存输入形状，例如 (4, 2) 或 (N, 4, 2)，用于最后恢复结构。
    original_shape = points.shape
    # 拉平成二维点集，形状从 (..., 2) 变为 (M, 2)，M 为总点数。
    points = points.reshape(-1, 2)
    # 构造齐次坐标的常数列，形状为 (M, 1)，用于表示平移项。
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    # 拼成齐次坐标，形状从 (M, 2) 变为 (M, 3)，每行是 [x, y, 1]。
    points_h = np.concatenate([points, ones], axis=1)
    # 右乘 matrix.T，使 (M, 3) @ (3, 2) 得到 (M, 2) 的变换后坐标。
    transformed = points_h @ np.asarray(matrix, dtype=np.float32).T
    # 转回 float32，并恢复为输入 points 的原始形状。
    return transformed.astype(np.float32).reshape(original_shape)


def get_scale_affine_matrix(scale_x: float, scale_y: float) -> np.ndarray:
    """构造坐标缩放的仿射矩阵。

    参数:
        scale_x (float): x 坐标缩放比例。
        scale_y (float): y 坐标缩放比例。

    返回:
        np.ndarray:
            形状为 ``(2, 3)`` 的仿射矩阵。
            输入点 ``(..., 2)`` 经过该矩阵变换后，输出形状仍为 ``(..., 2)``。
    """
    return np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)


def get_img_augment_affine_matrix(mode: int, img_shape) -> np.ndarray:
    """构造与 ``img_augment`` 的 8 种模式一致的标签仿射矩阵。

    参数:
        mode (int): ``img_augment`` 的增强模式，取值范围 ``[0, 7]``。
        img_shape: 原图形状，取前两维作为 ``(H, W)``。

    返回:
        np.ndarray:
            形状为 ``(2, 3)`` 的仿射矩阵。
            DOTA 标签 ``polygon`` 从 ``(4, 2)`` 变换后仍为 ``(4, 2)``。
    """
    height, width = img_shape[:2]
    if mode == 0:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    if mode == 1:
        return np.array([[0, 1, 0], [-1, 0, width - 1]], dtype=np.float32)
    if mode == 2:
        return np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
    if mode == 3:
        return np.array([[-1, 0, width - 1], [0, -1, height - 1]], dtype=np.float32)
    if mode == 4:
        return np.array([[-1, 0, width - 1], [0, 1, 0]], dtype=np.float32)
    if mode == 5:
        return np.array([[0, -1, height - 1], [1, 0, 0]], dtype=np.float32)
    if mode == 6:
        return np.array([[0, -1, height - 1], [-1, 0, width - 1]], dtype=np.float32)
    if mode == 7:
        return np.array([[1, 0, 0], [0, -1, height - 1]], dtype=np.float32)
    raise ValueError(f"mode must be in [0, 7], but got {mode}")


def get_img_rotate_affine_matrix(img_shape, angle, center=None, scale=1.0) -> np.ndarray:
    """构造与 ``img_rotate`` 一致的旋转仿射矩阵。

    参数:
        img_shape: 原图形状，取前两维作为 ``(H, W)``。
        angle (float): 旋转角度，正值表示逆时针旋转。
        center (tuple[int] | None): 旋转中心，默认为图像中心 ``(W // 2, H // 2)``。
        scale (float): 等比缩放因子。

    返回:
        np.ndarray:
            形状为 ``(2, 3)`` 的仿射矩阵。
            输入点 ``(..., 2)`` 经过该矩阵变换后，输出形状仍为 ``(..., 2)``。
    """
    height, width = img_shape[:2]
    if center is None:
        center = (width // 2, height // 2)
    return cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)


def transform_polygons_by_affine(polygons: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """使用仿射矩阵变换多边形数组。

    参数:
        polygons (np.ndarray):
            形状为 ``(4, 2)`` 或 ``(N, 4, 2)`` 的多边形坐标。
        matrix (np.ndarray):
            形状为 ``(2, 3)`` 的仿射矩阵。

    返回:
        np.ndarray: 形状与输入 ``polygons`` 一致。
    """
    return transform_points_by_affine(polygons, matrix)


def transform_dota_polygon_by_affine(polygon: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """使用仿射矩阵变换 DOTA 四点框。

    参数:
        polygon (np.ndarray): 形状为 ``(4, 2)`` 的多边形四个顶点。
        matrix (np.ndarray):
            形状为 ``(2, 3)`` 的仿射矩阵。
            可来自缩放、8 模式增强或任意角度旋转。

    返回:
        np.ndarray: 形状为 ``(4, 2)`` 的变换后顶点坐标。
    """
    return transform_polygons_by_affine(np.asarray(polygon, dtype=np.float32).reshape(4, 2), matrix)


def transform_dota_labels_by_affine(labels, matrix: np.ndarray):
    """使用仿射矩阵统一变换 DOTA 标签。

    参数:
        labels (list[dict]):
            DOTA 标注列表，长度为 ``N``。
            每个元素至少包含 ``polygon``，其形状为 ``(4, 2)``。
        matrix (np.ndarray):
            形状为 ``(2, 3)`` 的仿射矩阵。
            统一表示标签缩放、8 模式增强、任意角度旋转等坐标变换。

    返回:
        list[dict]:
            变换后的 DOTA 标注列表，长度仍为 ``N``。
            每个元素的 ``polygon`` 仍为 ``(4, 2)``，``class_name`` 和 ``difficult`` 等字段保持不变。
    """
    polygons = np.asarray([item["polygon"] for item in labels], dtype=np.float32).reshape(-1, 4, 2)
    transformed_polygons = transform_polygons_by_affine(polygons, matrix)
    out = []
    for item, polygon in zip(labels, transformed_polygons):
        new_item = dict(item)
        new_item["polygon"] = polygon
        out.append(new_item)
    return out


def img_augment_DOTA_label(labels, mode=0, img_shape=None):
    """对 DOTA 检测标签执行与 ``img_augment`` 一致的几何变换。

    DOTA 标签格式通常为:
        {
            "polygon": np.ndarray(shape=(4, 2), dtype=float32),
            "class_name": str,
            "difficult": str | int
        }

    参数:
        labels (list[dict]):
            DOTA 标注列表。每个元素对应一个目标，且 ``polygon`` 形状必须为 ``(4, 2)``。
        mode (int):
            与 ``img_augment`` 一致的增强模式，取值范围 ``[0, 7]``。
        img_shape (tuple[int, int] | None):
            原始图像尺寸 ``(H, W)``。坐标变换需要它来计算翻转与旋转后的坐标。

    返回:
        list[dict]: 变换后的 DOTA 标注列表。
            - 输入长度为 ``N``，输出长度仍为 ``N``；
            - 每个元素的 ``polygon`` 仍为 ``(4, 2)``；
            - ``class_name`` 和 ``difficult`` 保持不变。
    """
    matrix = get_img_augment_affine_matrix(mode, img_shape)
    return transform_dota_labels_by_affine(labels, matrix)




def img_rotate_DOTA_label(labels, img_shape, angle, center=None, scale=1.0):
    """对 DOTA 检测标签执行与 ``img_rotate`` 一致的旋转变换。

    DOTA 标签格式通常为:
        {
            "polygon": np.ndarray(shape=(4, 2), dtype=float32),
            "class_name": str,
            "difficult": str | int
        }

    参数:
        labels (list[dict]):
            DOTA 标注列表。每个元素对应一个目标，且 ``polygon`` 形状必须为 ``(4, 2)``。
        img_shape (tuple[int, int]):
            原始图像尺寸 ``(H, W)``。
        angle (float):
            旋转角度，正值表示逆时针旋转。
        center (tuple[int] | None):
            旋转中心。``None`` 时默认使用图像中心。
        scale (float):
            等比缩放因子，默认 ``1.0``。

    返回:
        list[dict]: 变换后的 DOTA 标注列表。
            - 输入长度为 ``N``，输出长度仍为 ``N``；
            - 每个元素的 ``polygon`` 仍为 ``(4, 2)``；
            - ``class_name`` 和 ``difficult`` 保持不变。
    """
    matrix = get_img_rotate_affine_matrix(img_shape, angle, center=center, scale=scale)
    return transform_dota_labels_by_affine(labels, matrix)


if __name__ == "__main__":
    pass
