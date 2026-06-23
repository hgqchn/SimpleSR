from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Iterable


def collect_dota_pairs(
    image_dir: str | Path,
    label_dir: str | Path,
    image_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    require_label: bool = True,
) -> list[tuple[Path, Path]]:
    """收集 DOTA 图像与标签配对。

    DOTA 通常使用同名文件配对，例如:
        images/P0000.png
        labelTxt/P0000.txt

    Args:
        image_dir: 原始图像目录。
        label_dir: 原始标签目录。
        image_suffixes: 允许的图像后缀。
        require_label: 是否要求每张图像必须存在对应标签。

    Returns:
        图像路径和标签路径组成的列表。
    """
    image_dir = Path(image_dir).expanduser().resolve()
    label_dir = Path(label_dir).expanduser().resolve()

    if not image_dir.is_dir():
        raise NotADirectoryError(f"图像目录不存在或不是目录: {image_dir}")
    if not label_dir.is_dir():
        raise NotADirectoryError(f"标签目录不存在或不是目录: {label_dir}")

    suffixes = {s.lower() for s in image_suffixes}
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path)
            if not require_label:
                pairs.append((image_path, label_path))

    if require_label and missing_labels:
        print(f"[Warning] 有 {len(missing_labels)} 张图像缺少标签，已跳过。")

    if len(pairs) == 0:
        raise RuntimeError(
            f"没有找到有效的 DOTA 图像-标签配对: image_dir={image_dir}, label_dir={label_dir}"
        )

    return pairs


def random_select_dota_samples(
    image_dir: str | Path,
    label_dir: str | Path,
    manifest_path: str | Path,
    num_samples: int,
    seed: int = 0,
    image_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    require_label: bool = True,
) -> list[tuple[Path, Path]]:
    """从 DOTA 数据集中随机挑选样本，并保存图像路径和标签路径。

    Args:
        image_dir: 原始图像目录。
        label_dir: 原始标签目录。
        manifest_path: 保存挑选结果的 CSV 文件路径。
        num_samples: 随机挑选的样本数量。
        seed: 随机种子，保证可复现。
        image_suffixes: 允许的图像后缀。
        require_label: 是否要求图像必须有对应标签。

    Returns:
        被挑选出的图像-标签路径列表。
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples 必须大于 0，但得到: {num_samples}")

    pairs = collect_dota_pairs(
        image_dir=image_dir,
        label_dir=label_dir,
        image_suffixes=image_suffixes,
        require_label=require_label,
    )

    if num_samples > len(pairs):
        raise ValueError(
            f"num_samples={num_samples} 大于可用样本数 {len(pairs)}"
        )

    rng = random.Random(seed)
    #从 pairs 这个列表里，随机抽取 num_samples 个元素，且不重复
    selected_pairs = rng.sample(pairs, num_samples)

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label_path", "name"])

        for image_path, label_path in selected_pairs:
            writer.writerow([
                str(image_path),
                str(label_path),
                image_path.stem,
            ])

    print(f"[Info] 已随机挑选 {len(selected_pairs)} 个样本")
    print(f"[Info] manifest 保存到: {manifest_path}")

    return selected_pairs


def read_dota_manifest(
    manifest_path: str | Path,
) -> list[tuple[Path, Path, str]]:
    """读取随机挑选得到的 manifest 文件。

    Args:
        manifest_path: CSV manifest 文件路径。

    Returns:
        image_path, label_path, name 组成的列表。
    """
    manifest_path = Path(manifest_path).expanduser().resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 文件不存在: {manifest_path}")

    samples: list[tuple[Path, Path, str]] = []

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_fields = {"image_path", "label_path", "name"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                f"manifest 文件字段错误，需要包含: {required_fields}，"
                f"当前字段: {reader.fieldnames}"
            )

        for row in reader:
            image_path = Path(row["image_path"]).expanduser().resolve()
            label_path = Path(row["label_path"]).expanduser().resolve()
            name = row["name"]
            samples.append((image_path, label_path, name))

    return samples


def copy_dota_samples_from_manifest(
    manifest_path: str | Path,
    output_image_dir: str | Path,
    output_label_dir: str | Path,
    overwrite: bool = False,
) -> None:
    """根据 manifest 文件复制 DOTA 图像和标签到新目录。

    Args:
        manifest_path: 由 random_select_dota_samples 生成的 CSV 文件。
        output_image_dir: 新数据集图像保存目录。
        output_label_dir: 新数据集标签保存目录。
        overwrite: 如果目标文件已存在，是否覆盖。
    """
    output_image_dir = Path(output_image_dir).expanduser().resolve()
    output_label_dir = Path(output_label_dir).expanduser().resolve()

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    samples = read_dota_manifest(manifest_path)

    for image_path, label_path, name in samples:
        if not image_path.is_file():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"标签文件不存在: {label_path}")

        dst_image_path = output_image_dir / image_path.name
        dst_label_path = output_label_dir / label_path.name

        if not overwrite:
            if dst_image_path.exists():
                raise FileExistsError(f"目标图像已存在: {dst_image_path}")
            if dst_label_path.exists():
                raise FileExistsError(f"目标标签已存在: {dst_label_path}")

        shutil.copy2(image_path, dst_image_path)
        shutil.copy2(label_path, dst_label_path)

    print(f"[Info] 已复制 {len(samples)} 个样本")
    print(f"[Info] 图像保存到: {output_image_dir}")
    print(f"[Info] 标签保存到: {output_label_dir}")


def make_dota_dataset(
    image_dir: str | Path,
    label_dir: str | Path,
    output_root: str | Path,
    num_samples: int,
    seed: int = 0,
    manifest_name: str = "selected_samples.csv",
    overwrite: bool = False,
) -> Path:
    """一步完成随机挑选和复制，制作 DOTA 数据集。

    输出结构:
        output_root/
        ├── images/
        ├── labelTxt/
        └── selected_samples.csv

    Args:
        image_dir: 原始 DOTA 图像目录。
        label_dir: 原始 DOTA 标签目录。
        output_root: 数据集输出根目录。
        num_samples: 随机挑选样本数量。
        seed: 随机种子。
        manifest_name: 保存路径列表的 CSV 文件名。
        overwrite: 是否覆盖已有文件。

    Returns:
        manifest 文件路径。
    """
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    output_image_dir = output_root / "images"
    output_label_dir = output_root / "labelTxt"
    manifest_path = output_root / manifest_name

    random_select_dota_samples(
        image_dir=image_dir,
        label_dir=label_dir,
        manifest_path=manifest_path,
        num_samples=num_samples,
        seed=seed,
    )

    copy_dota_samples_from_manifest(
        manifest_path=manifest_path,
        output_image_dir=output_image_dir,
        output_label_dir=output_label_dir,
        overwrite=overwrite,
    )

    return manifest_path


if __name__ == '__main__':
    dota_train=r'D:\Data\RemoteSensing\DOTA_crop\train_crop_256\images'
    dota_train_label=r'D:\Data\RemoteSensing\DOTA_crop\train_crop_256\labelTxt'
    dota_val=r'D:\Data\RemoteSensing\DOTA_crop\val_crop_256\images'
    dota_val_label=r'D:\Data\RemoteSensing\DOTA_crop\val_crop_256\labelTxt'
    #
    # output_dir_train=r'../DOTA_crop_dataset/dota_samples/train'
    # output_dir_val=r'../DOTA_crop_dataset/dota_samples/val'
    #
    # make_dota_dataset(
    #     image_dir=dota_train,
    #     label_dir=dota_train_label,
    #     output_root=output_dir_train,
    #     num_samples=400,
    #     seed=0,
    #     manifest_name="selected_samples_train.csv",
    # )
    #
    # make_dota_dataset(
    #     image_dir=dota_val,
    #     label_dir=dota_val_label,
    #     output_root=output_dir_val,
    #     num_samples=100,
    #     seed=0,
    #     manifest_name="selected_samples_val.csv",
    # )

    output_dir_val=r'../dataset/DOTA_crop_dataset/dota_samples/val_mini'

    make_dota_dataset(
        image_dir=dota_val,
        label_dir=dota_val_label,
        output_root=output_dir_val,
        num_samples=10,
        seed=0,
        manifest_name="selected_samples_val.csv",
    )