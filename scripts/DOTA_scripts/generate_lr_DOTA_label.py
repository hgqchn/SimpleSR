from pathlib import Path
from typing import Optional

from tqdm import tqdm

from DOTA_utils.label_transform_utils import scale_dota_label_file


def scale_dota_label_folder(
    src_label_dir: str,
    dst_label_dir: str,
    scale: int = 4,
    recursive: bool = True,
    min_box_size: Optional[float] = None,
) -> None:
    """批量缩放 DOTA 标签文件夹。

    参数:
        src_label_dir: HR 标签文件夹。
        dst_label_dir: LR 标签输出文件夹。
        scale: HR 到 LR 的倍率。例如 ``scale=4`` 表示坐标除以 4。
        recursive: 是否递归处理子文件夹。
        min_box_size: 可选，过滤缩放后太小的目标。
    """
    src_label_dir = Path(src_label_dir)
    dst_label_dir = Path(dst_label_dir)

    label_paths = sorted(src_label_dir.rglob("*.txt") if recursive else src_label_dir.glob("*.txt"))
    scale_x = 1.0 / scale
    scale_y = 1.0 / scale

    for src_label_path in tqdm(label_paths, desc="Scaling DOTA labels"):
        rel_path = src_label_path.relative_to(src_label_dir)
        dst_label_path = dst_label_dir / rel_path
        scale_dota_label_file(
            src_label_path=src_label_path,
            dst_label_path=dst_label_path,
            scale_x=scale_x,
            scale_y=scale_y,
            min_box_size=min_box_size,
        )

    print(f"完成，LR 标签保存到: {dst_label_dir}")


if __name__ == '__main__':
    train_label_dir = r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train\labelTxt"
    train_lr_label_dir = r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train_lrx4\labelTxt"

    val_label_dir = r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\val\labelTxt"
    val_lr_label_dir = r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\val_lrx4\labelTxt"

    scale_dota_label_folder(
        train_label_dir,
        train_lr_label_dir,
        scale=4,
        recursive=True,
        min_box_size=2,
    )

    scale_dota_label_folder(
        val_label_dir,
        val_lr_label_dir,
        scale=4,
        recursive=True,
        min_box_size=2,
    )
