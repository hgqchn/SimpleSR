from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


LabelKeepPolicy = Literal["inside", "center", "intersect", "all"]


def find_non_black_bbox(
    img: np.ndarray,
    black_threshold: int = 5,
    margin: int = 0,
    align_to: int | None = None,
) -> tuple[int, int, int, int]:
    """检测图像非黑区域的外接矩形。

    Args:
        img: 输入图像，shape 为 (H, W) 或 (H, W, C)。
        black_threshold: 黑色阈值。像素强度小于等于该值视为黑色。
        margin: 在检测到的有效区域外额外保留的边界像素数。
        align_to: 若不为 None，则让裁剪后宽高可以被该值整除。
                  例如超分 scale=4 时，可设置 align_to=4。

    Returns:
        bbox: (x_min, y_min, x_max, y_max)，其中 x_max/y_max 为右开边界。
    """
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        # 用通道均值判断是否接近黑色。
        gray = img.mean(axis=2)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    h, w = gray.shape[:2]

    valid_mask = gray > black_threshold
    ys, xs = np.where(valid_mask)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No non-black region found in image.")

    x_min = max(int(xs.min()) - margin, 0)
    x_max = min(int(xs.max()) + 1 + margin, w)
    y_min = max(int(ys.min()) - margin, 0)
    y_max = min(int(ys.max()) + 1 + margin, h)

    if align_to is not None:
        if align_to <= 0:
            raise ValueError(f"align_to must be positive, but got {align_to}")

        crop_w = x_max - x_min
        crop_h = y_max - y_min

        new_w = crop_w // align_to * align_to
        new_h = crop_h // align_to * align_to

        if new_w <= 0 or new_h <= 0:
            raise ValueError(
                f"Aligned crop size is invalid: ({new_h}, {new_w}). "
                f"Original crop size: ({crop_h}, {crop_w}), align_to={align_to}"
            )

        # 这里采用从右边和下边收缩，保持左上角不变，标签平移更简单。
        x_max = x_min + new_w
        y_max = y_min + new_h

    return x_min, y_min, x_max, y_max


def crop_image_by_bbox(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """根据 bbox 裁剪图像。"""
    x_min, y_min, x_max, y_max = bbox
    return img[y_min:y_max, x_min:x_max, ...]


def _is_number(text: str) -> bool:
    """判断字符串是否可以转成数字。"""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _format_coord(value: float) -> str:
    """格式化 DOTA 坐标。"""
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def _polygon_keep_by_policy(
    points: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    policy: LabelKeepPolicy,
) -> bool:
    """判断一个 DOTA 多边形框是否保留。"""
    if policy == "all":
        return True

    x_min, y_min, x_max, y_max = crop_bbox

    xs = points[:, 0]
    ys = points[:, 1]

    if policy == "inside":
        return bool(
            np.all(xs >= x_min)
            and np.all(xs < x_max)
            and np.all(ys >= y_min)
            and np.all(ys < y_max)
        )

    if policy == "center":
        cx = float(xs.mean())
        cy = float(ys.mean())
        return bool(x_min <= cx < x_max and y_min <= cy < y_max)

    if policy == "intersect":
        poly_x_min = float(xs.min())
        poly_x_max = float(xs.max())
        poly_y_min = float(ys.min())
        poly_y_max = float(ys.max())

        no_intersection = (
            poly_x_max < x_min
            or poly_x_min >= x_max
            or poly_y_max < y_min
            or poly_y_min >= y_max
        )
        return not no_intersection

    raise ValueError(f"Unsupported label keep policy: {policy}")


def shift_dota_label_by_crop(
    src_label_path: str | os.PathLike[str],
    dst_label_path: str | os.PathLike[str],
    crop_bbox: tuple[int, int, int, int],
    keep_policy: LabelKeepPolicy = "inside",
    clip_coords: bool = False,
    encoding: str = "utf-8",
) -> int:
    """根据原图裁剪框，平移 DOTA 标签坐标并保存。

    DOTA 标签通常格式:
        x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult

    Args:
        src_label_path: 原始 DOTA labelTxt 文件路径。
        dst_label_path: 输出 labelTxt 文件路径。
        crop_bbox: 图像裁剪框，格式为 (x_min, y_min, x_max, y_max)。
        keep_policy: 标签保留策略。
            - "inside": 只保留四个点都在裁剪区域内的目标。
            - "center": 只保留中心点在裁剪区域内的目标。
            - "intersect": 只要目标 bbox 与裁剪区域有交集就保留。
            - "all": 全部保留，只做坐标平移。
        clip_coords: 是否把平移后的坐标裁剪到新图范围内。
                     注意：对旋转框来说，简单 clip 可能改变框形状，默认 False。
        encoding: 标签文件编码。

    Returns:
        kept_count: 保留下来的目标数量。
    """
    src_label_path = Path(src_label_path)
    dst_label_path = Path(dst_label_path)
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)

    x_min, y_min, x_max, y_max = crop_bbox
    new_w = x_max - x_min
    new_h = y_max - y_min

    if not src_label_path.exists():
        # 没有标签时，创建一个空标签文件。
        dst_label_path.write_text("", encoding=encoding)
        return 0

    kept_lines: list[str] = []
    kept_count = 0

    with src_label_path.open("r", encoding=encoding) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # DOTA 前两行可能是:
        # imagesource:xxx
        # gsd:xxx
        # 这类行不是目标标注，直接保留。
        if len(parts) < 9 or not all(_is_number(v) for v in parts[:8]):
            kept_lines.append(line)
            continue

        coords = np.array([float(v) for v in parts[:8]], dtype=np.float32)
        points = coords.reshape(4, 2)

        if not _polygon_keep_by_policy(points, crop_bbox, keep_policy):
            continue

        # 坐标平移到裁剪后图像坐标系。
        points[:, 0] -= x_min
        points[:, 1] -= y_min

        if clip_coords:
            points[:, 0] = np.clip(points[:, 0], 0, new_w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, new_h - 1)

        new_coords = points.reshape(-1).tolist()
        new_coord_strs = [_format_coord(v) for v in new_coords]

        # class_name 和 difficult 等剩余字段原样保留。
        new_line = " ".join(new_coord_strs + parts[8:])
        kept_lines.append(new_line)
        kept_count += 1

    with dst_label_path.open("w", encoding=encoding) as f:
        for line in kept_lines:
            f.write(line + "\n")

    return kept_count


def crop_dota_image_and_label_remove_black_border(
    src_img_path: str | os.PathLike[str],
    src_label_path: str | os.PathLike[str],
    dst_img_path: str | os.PathLike[str],
    dst_label_path: str | os.PathLike[str],
    black_threshold: int = 5,
    margin: int = 0,
    align_to: int | None = None,
    keep_policy: LabelKeepPolicy = "inside",
    clip_coords: bool = False,
) -> tuple[tuple[int, int, int, int], int]:
    """对单张 DOTA 大图去黑边裁剪，并同步处理标签。

    Args:
        src_img_path: 原始图像路径。
        src_label_path: 原始标签路径。
        dst_img_path: 裁剪后图像保存路径。
        dst_label_path: 裁剪后标签保存路径。
        black_threshold: 黑边阈值。
        margin: 有效区域外额外保留边界。
        align_to: 裁剪后宽高对齐到该倍数，例如 scale=4。
        keep_policy: 标签保留策略。
        clip_coords: 是否 clip 平移后的坐标。

    Returns:
        bbox: 原图坐标系中的裁剪框。
        kept_count: 保留的目标数量。
    """
    src_img_path = Path(src_img_path)
    dst_img_path = Path(dst_img_path)
    dst_label_path = Path(dst_label_path)

    img = cv2.imread(str(src_img_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Failed to read image: {src_img_path}")

    bbox = find_non_black_bbox(
        img,
        black_threshold=black_threshold,
        margin=margin,
        align_to=align_to,
    )

    cropped_img = crop_image_by_bbox(img, bbox)

    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst_img_path), cropped_img)

    if not ok:
        raise IOError(f"Failed to write image: {dst_img_path}")

    kept_count = shift_dota_label_by_crop(
        src_label_path=src_label_path,
        dst_label_path=dst_label_path,
        crop_bbox=bbox,
        keep_policy=keep_policy,
        clip_coords=clip_coords,
    )

    return bbox, kept_count

def remove_black_border_for_dota_dataset(
    src_img_dir: str | os.PathLike[str],
    src_label_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    image_suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    label_name: str = "labelTxt",
    black_threshold: int = 5,
    margin: int = 0,
    align_to: int | None = None,
    keep_policy: LabelKeepPolicy = "inside",
    clip_coords: bool = False,
) -> None:
    """批量对 DOTA 原始大图去黑边裁剪，并同步处理标签。

    输出结构:
        out_dir/
        ├── images/
        └── labelTxt/
    """
    src_img_dir = Path(src_img_dir).expanduser().resolve()
    src_label_dir = Path(src_label_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    out_img_dir = out_dir / "images"
    out_label_dir = out_dir / label_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    suffixes = {s.lower() for s in image_suffixes}

    image_paths = sorted(
        p for p in src_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )

    print(f"Found {len(image_paths)} images.")
    print(f"Output image dir: {out_img_dir}")
    print(f"Output label dir: {out_label_dir}")

    for idx, img_path in enumerate(image_paths, start=1):
        label_path = src_label_dir / f"{img_path.stem}.txt"

        dst_img_path = out_img_dir / img_path.name
        dst_label_path = out_label_dir / f"{img_path.stem}.txt"

        print(f"[{idx}/{len(image_paths)}] Processing {img_path.name}")

        bbox, kept_count = crop_dota_image_and_label_remove_black_border(
            src_img_path=img_path,
            src_label_path=label_path,
            dst_img_path=dst_img_path,
            dst_label_path=dst_label_path,
            black_threshold=black_threshold,
            margin=margin,
            align_to=align_to,
            keep_policy=keep_policy,
            clip_coords=clip_coords,
        )

        print(f"  bbox={bbox}, kept objects={kept_count}")

    print("Done.")


if __name__ == "__main__":


    pass