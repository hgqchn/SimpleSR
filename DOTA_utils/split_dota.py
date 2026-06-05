from pathlib import Path
from typing import List, Dict, Tuple
import argparse

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed


from DOTA_utils.utils import parse_dota_label,collect_images,write_dota_label





def get_start_positions(length: int, crop_size: int, stride: int) -> List[int]:
    """
    生成滑窗裁剪的起始位置列表。

    """
    if length <= crop_size:
        print(f"Warning: image length {length} is smaller than or equal to crop size {crop_size}.")
        return [0]

    positions = list(range(0, length - crop_size + 1, stride))

    # last = length - crop_size
    # if positions[-1] != last:
    #     positions.append(last)

    return positions


def is_polygon_inside_crop(
    polygon: np.ndarray,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    eps: float = 1e-6,
) -> bool:
    """
    检查多边形的所有顶点是否都位于当前裁剪窗口内部。
    """
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    xs = polygon[:, 0]
    ys = polygon[:, 1]

    inside_x = np.all((xs >= x0 - eps) & (xs <= x1 + eps))
    inside_y = np.all((ys >= y0 - eps) & (ys <= y1 + eps))

    return bool(inside_x and inside_y)


def convert_labels_for_crop(
    objects: List[Dict],
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    keep_policy: str = "inside",
) -> List[Dict]:
    """
    将原始 DOTA 标注转换为某个裁剪子图对应的标注。

    keep_policy 含义：
        inside:
            仅保留四个顶点都完整落在裁剪区域内的目标。
            这是更推荐的策略，因为可以保持 DOTA 旋转框标注干净完整。

        center:
            保留中心点位于裁剪区域内的目标。
            对于较大的目标，这种方式可能生成超出子图边界的坐标。
            只有当后续处理流程能够接受这类标注时才建议使用。
    """
    crop_objects = []

    for obj in objects:
        polygon = obj["polygon"]

        if keep_policy == "inside":
            keep = is_polygon_inside_crop(
                polygon=polygon,
                x0=x0,
                y0=y0,
                crop_w=crop_w,
                crop_h=crop_h,
            )

        elif keep_policy == "center":
            center = polygon.mean(axis=0)
            cx, cy = center
            keep = (
                x0 <= cx <= x0 + crop_w
                and y0 <= cy <= y0 + crop_h
            )

        else:
            raise ValueError(f"Unknown keep_policy: {keep_policy}")

        if not keep:
            continue

        new_polygon = polygon.copy()
        new_polygon[:, 0] -= x0
        new_polygon[:, 1] -= y0

        crop_objects.append(
            {
                "polygon": new_polygon,
                "class_name": obj["class_name"],
                "difficult": obj["difficult"],
            }
        )

    return crop_objects


def crop_image_fixed_size(
    image: Image.Image,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
) -> Image.Image:
    """
    按固定输出尺寸裁剪图像。

    如果图像尺寸小于目标裁剪尺寸，则对缺失区域进行填充。
    """
    w, h = image.size

    x1 = min(x0 + crop_w, w)
    y1 = min(y0 + crop_h, h)

    crop = image.crop((x0, y0, x1, y1))

    return crop

def split_single_image(
    image_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_label_dir: Path,
    crop_size: int,
    stride: int,
    keep_policy: str = "inside",
    drop_empty: bool = False,
    image_format: str = "png",
) -> None:
    """
    对单张 DOTA 图像及其对应标注文件进行切分。
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    objects = parse_dota_label(label_path)

    x_positions = get_start_positions(w, crop_size, stride)
    y_positions = get_start_positions(h, crop_size, stride)

    image_stem = image_path.stem

    for y0 in y_positions:
        for x0 in x_positions:
            crop_objects = convert_labels_for_crop(
                objects=objects,
                x0=x0,
                y0=y0,
                crop_w=crop_size,
                crop_h=crop_size,
                keep_policy=keep_policy,
            )

            if drop_empty and len(crop_objects) == 0:
                continue

            crop = crop_image_fixed_size(
                image=image,
                x0=x0,
                y0=y0,
                crop_w=crop_size,
                crop_h=crop_size,
            )

            out_img_dir.mkdir(exist_ok=True, parents=True)
            out_label_dir.mkdir(exist_ok=True, parents=True)

            sub_name = f"{image_stem}_x{x0:04d}_y{y0:04d}"
            sub_img_path = out_img_dir / f"{sub_name}.{image_format}"
            sub_label_path = out_label_dir / f"{sub_name}.txt"

            crop.save(sub_img_path)
            write_dota_label(sub_label_path, crop_objects)


def _split_single_image_worker(args: tuple) -> str:
    """多进程 worker：处理单张 DOTA 图像。"""
    (
        image_path,
        src_label_dir,
        out_img_dir,
        out_label_dir,
        crop_size,
        stride,
        keep_policy,
        drop_empty,
        image_format,
    ) = args

    image_path = Path(image_path)
    src_label_dir = Path(src_label_dir)
    out_img_dir = Path(out_img_dir)
    out_label_dir = Path(out_label_dir)

    label_path = src_label_dir / f"{image_path.stem}.txt"

    split_single_image(
        image_path=image_path,
        label_path=label_path,
        out_img_dir=out_img_dir,
        out_label_dir=out_label_dir,
        crop_size=crop_size,
        stride=stride,
        keep_policy=keep_policy,
        drop_empty=drop_empty,
        image_format=image_format,
    )

    return image_path.name


def split_dota_dataset(
    src_img_dir: str | os.PathLike[str],
    src_label_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    crop_size: int = 512,
    stride: int = 512,
    keep_policy: str = "inside",
    drop_empty: bool = False,
    image_format: str = "png",
    label_name: str = "labelTxt",
    num_workers: int = 1,
) -> None:
    """
    切分一个 DOTA 风格的数据集，支持单进程和多进程。

    输入:
        src_img_dir:
            原始 DOTA 图像目录。

        src_label_dir:
            原始 DOTA 标注目录。

    输出:
        out_dir/images:
            裁剪后的子图。

        out_dir/{label_name}:
            裁剪后的标注。

    Args:
        src_img_dir: 原始图像目录。
        src_label_dir: 原始标签目录。
        out_dir: 输出根目录。
        crop_size: 裁剪 patch 大小。
        stride: 滑窗步长。
        keep_policy: 标注保留策略，例如 "inside"。
        drop_empty: 是否丢弃无目标 patch。
        image_format: 输出图像格式，例如 "png"。
        label_name: 输出标签文件夹名称。
        num_workers: 进程数。1 表示单进程；大于 1 时启用多进程。
    """
    src_img_dir = Path(src_img_dir).expanduser().resolve()
    src_label_dir = Path(src_label_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    if not src_img_dir.is_dir():
        raise NotADirectoryError(f"src_img_dir 不存在或不是目录: {src_img_dir}")
    if not src_label_dir.is_dir():
        raise NotADirectoryError(f"src_label_dir 不存在或不是目录: {src_label_dir}")

    out_img_dir = out_dir / "images"
    out_label_dir = out_dir / label_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(src_img_dir)

    print(f"Found {len(image_paths)} images.")
    print(f"Output image dir: {out_img_dir}")
    print(f"Output label dir: {out_label_dir}")
    print(f"Crop size: {crop_size}")
    print(f"Stride: {stride}")
    print(f"Keep policy: {keep_policy}")
    print(f"Drop empty patches: {drop_empty}")
    print(f"Image format: {image_format}")
    print(f"Num workers: {num_workers}")

    if len(image_paths) == 0:
        print("No images found. Done.")
        return

    if num_workers <= 1:
        for idx, image_path in enumerate(image_paths, start=1):
            label_path = src_label_dir / f"{image_path.stem}.txt"

            print(f"[{idx}/{len(image_paths)}] Processing {image_path.name}")

            split_single_image(
                image_path=image_path,
                label_path=label_path,
                out_img_dir=out_img_dir,
                out_label_dir=out_label_dir,
                crop_size=crop_size,
                stride=stride,
                keep_policy=keep_policy,
                drop_empty=drop_empty,
                image_format=image_format,
            )

    else:
        tasks = [
            (
                image_path,
                src_label_dir,
                out_img_dir,
                out_label_dir,
                crop_size,
                stride,
                keep_policy,
                drop_empty,
                image_format,
            )
            for image_path in image_paths
        ]

        max_workers = min(num_workers, os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_split_single_image_worker, task)
                for task in tasks
            ]

            progress_bar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Splitting images",
                unit="image",
            )

            for future in progress_bar:
                try:
                    image_name = future.result()
                    progress_bar.set_postfix_str(image_name)
                except Exception as e:
                    progress_bar.close()
                    raise RuntimeError(f"Failed to split image: {e}") from e

    print("Done.")




def main():
    parser = argparse.ArgumentParser(
        description="Split DOTA images and labels into fixed-size patches."
    )

    parser.add_argument(
        "--src-img-dir",
        type=str,
        required=True,
        help="Path to original DOTA images directory.",
    )

    parser.add_argument(
        "--src-label-dir",
        type=str,
        required=True,
        help="Path to original DOTA labelTxt directory.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path to output cropped dataset directory.",
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="Crop size of output patches.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding window stride.",
    )

    parser.add_argument(
        "--keep-policy",
        type=str,
        default="inside",
        choices=["inside", "center"],
        help="Policy for keeping objects in cropped labels.",
    )

    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop patches without any objects.",
    )

    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of worker processes.",
    )

    args = parser.parse_args()

    split_dota_dataset(
        src_img_dir=args.src_img_dir,
        src_label_dir=args.src_label_dir,
        out_dir=args.out_dir,
        crop_size=args.crop_size,
        stride=args.stride,
        keep_policy=args.keep_policy,
        drop_empty=args.drop_empty,
        image_format=args.image_format,
        num_workers=args.num_workers,
    )
# 原始数据裁剪为256x256
# python split_dota.py --src-img-dir D:\Data\RemoteSensing\DOTA\train\images --src-label-dir D:\Data\RemoteSensing\DOTA\train\labelTxt-v1.5\DOTA-v1.5_train --out-dir D:\Data\RemoteSensing\DOTA_crop\train_crop_256 --crop-size 256 --stride 256 --keep-policy inside --drop-empty --image-format png
# python split_dota.py --src-img-dir D:\Data\RemoteSensing\DOTA\val\images --src-label-dir D:\Data\RemoteSensing\DOTA\val\labelTxt-v1.5\DOTA-v1.5_val --out-dir D:\Data\RemoteSensing\DOTA_crop\val_crop_256 --crop-size 256 --stride 256 --keep-policy inside --drop-empty --image-format png


# python split_dota.py --src-img-dir D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train\images --src-label-dir D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train\labelTxt --out-dir D:\codes\My_SR_new\DOTA_crop_dataset\train_crop_256 --crop-size 256 --stride 256 --keep-policy inside --drop-empty --image-format png
# python split_dota.py --src-img-dir D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\val\images --src-label-dir D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\val\labelTxt --out-dir D:\codes\My_SR_new\DOTA_crop_dataset\val_crop_256 --crop-size 256 --stride 256 --keep-policy inside --drop-empty --image-format png


if __name__ == "__main__":
    main()
    # 单图测试
    # image_path = r"D:\Data\RemoteSensing\DOTA\train\images\P0005.png"
    # label_path = r"D:\Data\RemoteSensing\DOTA\train\labelTxt-v1.5\DOTA-v1.5_train\P0005.txt"
    #
    # split_single_image(
    #     image_path=Path(image_path),
    #     label_path=Path(label_path),
    #     out_img_dir=Path("./DOTA_test_crop/images"),
    #     out_label_dir=Path("./DOTA_test_crop/labelTxt"),
    #     crop_size=256,
    #     stride=256,
    #     keep_policy="inside",
    #     drop_empty=False,
    #     image_format="png",
    # )



    pass
