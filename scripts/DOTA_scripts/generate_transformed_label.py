from pathlib import Path

from DOTA_utils.utils import (
    dota_objects_to_polygons,
    filter_small_dota_objects,
    parse_dota_label,
    update_dota_objects_polygons,
    write_dota_label,
)
from simplesr.utils.img_utils import (
    get_img_augment_affine_matrix,
    get_img_rotate_affine_matrix,
    get_scale_affine_matrix,
    transform_polygons_by_affine,
)


def transform_dota_objects(objects, matrix, min_box_size=None):
    """使用仿射矩阵变换内存中的 DOTA 标注对象。

    输入:
        objects: 长度为 N 的 list[dict]，每个 ``polygon`` 形状为 ``(4, 2)``。
        matrix: 形状为 ``(2, 3)`` 的仿射矩阵。
        min_box_size: None 或最小外接矩形宽高阈值。

    输出:
        list[dict]: 长度不超过 N，每个 ``polygon`` 仍为 ``(4, 2)``。
    """
    polygons = dota_objects_to_polygons(objects)
    polygons = transform_polygons_by_affine(polygons, matrix)
    objects = update_dota_objects_polygons(objects, polygons)
    return filter_small_dota_objects(objects, min_box_size=min_box_size)


def scale_dota_objects(objects, scale_x, scale_y, min_box_size=None):
    """缩放内存中的 DOTA 标注对象。"""
    matrix = get_scale_affine_matrix(scale_x, scale_y)
    return transform_dota_objects(objects, matrix, min_box_size=min_box_size)


def augment_dota_objects(objects, img_shape, mode):
    """生成与 ``img_augment`` 对齐的 DOTA 标注对象。"""
    matrix = get_img_augment_affine_matrix(mode, img_shape)
    return transform_dota_objects(objects, matrix)


def rotate_dota_objects(objects, img_shape, angle, center=None, scale=1.0):
    """生成与 ``img_rotate`` 对齐的 DOTA 标注对象。"""
    matrix = get_img_rotate_affine_matrix(img_shape, angle, center=center, scale=scale)
    return transform_dota_objects(objects, matrix)


def transform_dota_label_file(src_label_path, dst_label_path, matrix, min_box_size=None):
    """读取 DOTA 标签文件，应用仿射矩阵后写出新标签。

    输入:
        src_label_path: 原始标签路径。
        dst_label_path: 输出标签路径。
        matrix: 形状为 ``(2, 3)`` 的仿射矩阵。
        min_box_size: None 或最小外接矩形宽高阈值。

    输出:
        写出的 DOTA 标签仍为 ``x1 y1 ... x4 y4 class_name difficult``。
    """
    src_label_path = Path(src_label_path)
    dst_label_path = Path(dst_label_path)

    if not src_label_path.exists():
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)
        dst_label_path.write_text("", encoding="utf-8")
        return

    objects = parse_dota_label(src_label_path)
    transformed_objects = transform_dota_objects(objects, matrix, min_box_size=min_box_size)
    write_dota_label(dst_label_path, transformed_objects)


def scale_dota_label_file(src_label_path, dst_label_path, scale_x, scale_y, min_box_size=None):
    """生成缩放图像对应的 DOTA 标签文件。"""
    matrix = get_scale_affine_matrix(scale_x, scale_y)
    transform_dota_label_file(
        src_label_path,
        dst_label_path,
        matrix,
        min_box_size=min_box_size,
    )


def augment_dota_label_file(src_label_path, dst_label_path, img_shape, mode):
    """生成与 ``img_augment`` 后图像对应的 DOTA 标签文件。"""
    matrix = get_img_augment_affine_matrix(mode, img_shape)
    transform_dota_label_file(src_label_path, dst_label_path, matrix)


def rotate_dota_label_file(src_label_path, dst_label_path, img_shape, angle, center=None, scale=1.0):
    """生成与 ``img_rotate`` 后图像对应的 DOTA 标签文件。"""
    matrix = get_img_rotate_affine_matrix(img_shape, angle, center=center, scale=scale)
    transform_dota_label_file(src_label_path, dst_label_path, matrix)


if __name__ == "__main__":
    pass
