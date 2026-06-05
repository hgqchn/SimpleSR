from pathlib import Path
import numpy as np
from typing import List, Dict

IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


DOTA_CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
    "container-crane",
    "airport",
    "helipad",
]

def parse_dota_label(label_path):
    """
    Parse DOTA label file.

    DOTA label format:
        x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult

    Some files may contain header lines:
        imagesource:xxx
        gsd:xxx
    """
    label_path = Path(label_path)
    objects = []

    if not label_path.exists():
        raise FileNotFoundError(f"Warning: label file not found: {label_path}")


    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("imagesource") or line.startswith("gsd"):
                continue

            parts = line.split()

            if len(parts) < 9:
                print(f"Skip invalid line: {line}")
                continue

            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                print(f"Skip invalid coordinates: {line}")
                continue

            polygon = np.array(coords, dtype=np.float32).reshape(4, 2)

            class_name = parts[8]

            if len(parts) >= 10:
                difficult = parts[9]
            else:
                difficult = "0"

            objects.append(
                {
                    "polygon": polygon,
                    "class_name": class_name,
                    "difficult": difficult,
                }
            )

    return objects

def collect_images(img_dir: Path) -> List[Path]:
    """
    从目录中收集所有图像文件。
    """
    image_paths = []

    for suffix in IMAGE_SUFFIXES:
        image_paths.extend(img_dir.glob(f"*{suffix}"))
        image_paths.extend(img_dir.glob(f"*{suffix.upper()}"))

    image_paths = sorted(set(image_paths))
    return image_paths

def format_number(x: float) -> str:
    """
    格式化坐标数值。

    如果该坐标接近整数，则按整数写出。
    否则保留两位小数。
    """
    if abs(x - round(x)) < 1e-4:
        return str(int(round(x)))
    return f"{x:.2f}"


def write_dota_label(label_path: Path|str, objects: List[Dict]) -> None:
    """
    写出裁剪后的 DOTA 标注文件。
    """
    if isinstance(label_path, str):
        label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with label_path.open("w", encoding="utf-8") as f:
        f.write("imagesource:GoogleEarth\n")
        f.write("gsd:None\n")

        for obj in objects:
            polygon = obj["polygon"].reshape(-1)
            class_name = obj["class_name"]
            difficult = obj["difficult"]

            coord_str = " ".join(format_number(float(v)) for v in polygon)
            line = f"{coord_str} {class_name} {difficult}\n"
            f.write(line)


def dota_objects_to_polygons(objects: List[Dict]) -> np.ndarray:
    """将 DOTA 标注对象转换为 polygon 数组。

    输入:
        objects: 长度为 N 的 list[dict]，每个对象的 ``polygon`` 形状为 ``(4, 2)``。

    输出:
        np.ndarray: 形状为 ``(N, 4, 2)`` 的四点框数组。
    """
    return np.asarray([obj["polygon"] for obj in objects], dtype=np.float32).reshape(-1, 4, 2)


def update_dota_objects_polygons(objects: List[Dict], polygons: np.ndarray) -> List[Dict]:
    """将 polygon 数组写回 DOTA 标注对象。

    输入:
        objects: 长度为 N 的 list[dict]。
        polygons: 形状为 ``(N, 4, 2)`` 的四点框数组。

    输出:
        list[dict]: 长度仍为 N，``polygon`` 已替换，其它字段保持不变。
    """
    polygons = np.asarray(polygons, dtype=np.float32).reshape(-1, 4, 2)
    new_objects = []
    for obj, polygon in zip(objects, polygons):
        new_obj = dict(obj)
        new_obj["polygon"] = polygon
        new_objects.append(new_obj)
    return new_objects


def filter_small_dota_objects(objects: List[Dict], min_box_size=None) -> List[Dict]:
    """过滤外接矩形过小的 DOTA 目标。

    输入:
        objects: DOTA 标注列表，每个 ``polygon`` 形状为 ``(4, 2)``。
        min_box_size: None 或最小宽高阈值。

    输出:
        list[dict]: 过滤后的 DOTA 标注列表。
    """
    if min_box_size is None:
        return objects

    filtered = []
    for obj in objects:
        polygon = np.asarray(obj["polygon"], dtype=np.float32).reshape(4, 2)
        box_w = polygon[:, 0].max() - polygon[:, 0].min()
        box_h = polygon[:, 1].max() - polygon[:, 1].min()
        if box_w >= min_box_size and box_h >= min_box_size:
            filtered.append(obj)
    return filtered

if __name__ == '__main__':
    pass
