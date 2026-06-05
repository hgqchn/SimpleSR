import os

from DOTA_utils.split_dota import convert_labels_for_crop
from DOTA_utils.utils import parse_dota_label, write_dota_label
from scripts.DOTA_scripts.generate_transformed_label import (
    augment_dota_objects,
    rotate_dota_objects,
    scale_dota_objects,
)
from simplesr.utils.img_utils import (
    crop_img_by_shape,
    get_crop_start,
    img_augment,
    img_rotate,
    read_img_as_rgb_float,
    write_rgb_float_img,
)
from simplesr.utils.matlab_bicubic import bicubic
from simplesr.utils.misc import get_filename


def prepare_output_dirs(output_dir):
    """创建图像和标签输出目录。"""
    imgs_dir = os.path.join(output_dir, "imgs")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return imgs_dir, labels_dir


def generate_center_crop_sample(img, objects, crop_shape, mode="center"):
    """生成固定大小裁剪图像及对应 DOTA 标签。

    输入:
        img: RGB 图像数组，形状为 ``(H, W, C)``。
        objects: DOTA 标签列表，每个 ``polygon`` 形状为 ``(4, 2)``。
        crop_shape: ``(crop_h, crop_w)``。

    输出:
        cropped_img: 裁剪图像，形状为 ``(crop_h, crop_w, C)``。
        cropped_objects: 裁剪标签列表，每个 ``polygon`` 仍为 ``(4, 2)``。
    """
    crop_h, crop_w = crop_shape
    cropped_img = crop_img_by_shape(img, crop_shape, mode)
    y0, x0 = get_crop_start(
        height=img.shape[0],
        width=img.shape[1],
        crop_h=crop_h,
        crop_w=crop_w,
        mode=mode,
    )
    cropped_objects = convert_labels_for_crop(objects, x0, y0, crop_w, crop_h)
    return cropped_img, cropped_objects


def write_sample(img, objects, img_path, label_path):
    """写出一组图像和 DOTA 标签。"""
    write_rgb_float_img(img, img_path)
    write_dota_label(label_path, objects)


def generate_scaled_sample(img, objects, scale):
    """生成缩放图像和对应 DOTA 标签对象。"""
    scaled_img = bicubic(img, scale=scale)
    scaled_objects = scale_dota_objects(objects, scale_x=scale, scale_y=scale)
    return scaled_img, scaled_objects


def write_scaled_samples(img, objects, img_name, ext, imgs_dir, labels_dir):
    """写出缩小和放大的验证样本。"""
    for suffix, scale in (("lrx4", 1.0 / 4), ("upx4", 4)):
        scaled_img, scaled_objects = generate_scaled_sample(img, objects, scale)
        write_sample(
            scaled_img,
            scaled_objects,
            os.path.join(imgs_dir, f"{img_name}_{suffix}{ext}"),
            os.path.join(labels_dir, f"{img_name}_{suffix}.txt"),
        )


def write_augment_samples(img, objects, img_name, ext, imgs_dir, labels_dir, modes=range(8)):
    """写出 8 种 ``img_augment`` 图像及对应标签。"""
    for mode in modes:
        aug_img = img_augment(img, mode)
        aug_objects = augment_dota_objects(objects, img_shape=img.shape[:2], mode=mode)
        write_sample(
            aug_img,
            aug_objects,
            os.path.join(imgs_dir, f"{img_name}_mode{mode}{ext}"),
            os.path.join(labels_dir, f"{img_name}_mode{mode}.txt"),
        )


def write_rotate_samples(img, objects, img_name, ext, imgs_dir, labels_dir, angles):
    """写出任意角度旋转图像及对应标签。"""
    for angle in angles:
        rotated_img = img_rotate(img, angle)
        rotated_objects = rotate_dota_objects(objects, img_shape=img.shape[:2], angle=angle)
        write_sample(
            rotated_img,
            rotated_objects,
            os.path.join(imgs_dir, f"{img_name}_angle{angle}{ext}"),
            os.path.join(labels_dir, f"{img_name}_angle{angle}.txt"),
        )


# 生成经过不同变换的图像与对应标签，用于验证函数正确性。
if __name__ == '__main__':
    output_dir = r'./test_data'
    imgs_dir, labels_dir = prepare_output_dirs(output_dir)

    img_path = r"D:\Data\RemoteSensing\DOTAv1_yolo\images\val\P0003.jpg"
    label_path = r"D:\Data\RemoteSensing\DOTAv1_yolo\labels\val_original\P0003.txt"

    img = read_img_as_rgb_float(img_path)
    img_name, ext = get_filename(img_path, with_ext=True)
    objects = parse_dota_label(label_path)

    cropped_img, cropped_objects = generate_center_crop_sample(
        img,
        objects,
        crop_shape=(600, 600),
        mode="center",
    )
    write_sample(
        cropped_img,
        cropped_objects,
        os.path.join(imgs_dir, f"{img_name}_crop{ext}"),
        os.path.join(labels_dir, f"{img_name}_crop.txt"),
    )

    write_scaled_samples(cropped_img, cropped_objects, img_name, ext, imgs_dir, labels_dir)
    write_augment_samples(cropped_img, cropped_objects, img_name, ext, imgs_dir, labels_dir)

    angles = [i * 45 for i in range(1, int(360 / 45))]
    write_rotate_samples(cropped_img, cropped_objects, img_name, ext, imgs_dir, labels_dir, angles)
