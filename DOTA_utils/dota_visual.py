from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from DOTA_utils.utils import parse_dota_label,collect_images


def clamp_text_position(x, y, width, height, pad=2):
    """将文字锚点限制在图像范围内。"""
    x = min(max(float(x), pad), width - pad)
    y = min(max(float(y), pad), height - pad)
    return x, y


def plot_on_ax(
    ax,
    image,
    title,
    objects,
    show_difficult=True,
    linewidth=2,
    ):
    ax.imshow(image)
    ax.axis("off")
    if hasattr(image, "size") and isinstance(image.size, tuple):
        width, height = image.size
    else:
        height, width = image.shape[:2]

    for obj in objects:
        polygon = obj["polygon"]
        class_name = obj["class_name"]
        difficult = obj["difficult"]

        patch = Polygon(
            polygon,
            closed=True,
            fill=False,
            edgecolor="red",
            linewidth=linewidth,
        )
        ax.add_patch(patch)

        x_text, y_text = polygon[0]
        x_text, y_text = clamp_text_position(x_text, y_text, width, height)
        ha = "right" if x_text > width * 0.75 else "left"
        va = "top" if y_text > height * 0.75 else "bottom"

        if show_difficult:
            text = f"{class_name} | d={difficult}"
        else:
            text = class_name

        ax.text(
            x_text,
            y_text,
            text,
            fontsize=8,
            color="yellow",
            ha=ha,
            va=va,
            bbox=dict(
                facecolor="black",
                alpha=0.6,
                edgecolor="none",
                pad=1,
            ),
        )

    ax.set_title(f"{title} | {len(objects)} objects", fontsize=10)

def visualize_dota_sample(
    image_path,
    label_path,
    save_path=None,
    show_difficult=True,
    linewidth=2,
    figsize=(8, 8),
    show=False,
):
    """
    Visualize one DOTA image and its oriented bounding box annotations.

    Args:
        image_path: path to image, e.g. P0001.png
        label_path: path to label txt, e.g. P0001.txt
        save_path: optional path to save visualization result
        show_difficult: whether to show difficult flag in text
        linewidth: polygon line width
        figsize: matplotlib figure size
        show: whether to show the figure
    """
    image_path = Path(image_path)
    label_path = Path(label_path)

    image = Image.open(image_path).convert("RGB")
    objects = parse_dota_label(label_path)

    img_name=image_path.stem
    fig, ax = plt.subplots(figsize=figsize)

    plot_on_ax(
        ax=ax,
        image=image,
        title=img_name,
        objects=objects,
        show_difficult=show_difficult,
        linewidth=linewidth,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.05,
        )

        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def visualize_dota_objects(
    image,
    objects,
    title="DOTA objects",
    save_path=None,
    show_difficult=True,
    linewidth=2,
    figsize=(8, 8),
    show=False,
):
    """直接可视化内存中的 DOTA 标注对象。

    参数:
        image: PIL.Image 或可被 ``imshow`` 显示的图像对象。
        objects: DOTA 标签列表，每个 ``polygon`` 形状为 ``(4, 2)``。
        title: 图像标题。
        save_path: 可选的可视化结果保存路径。
        show_difficult: 是否显示 difficult 字段。
        linewidth: 多边形线宽。
        figsize: matplotlib 图像尺寸。
        show: 是否弹出显示窗口。
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_on_ax(
        ax=ax,
        image=image,
        title=title,
        objects=objects,
        show_difficult=show_difficult,
        linewidth=linewidth,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)



def visualize_dota_dir(
    image_dir,
    label_dir,
    save_dir=None,
    show_difficult=False,
    linewidth=2,
    figsize=(8, 8),
    show=False,
):
    """
    Batch visualize DOTA images and labels.

    Args:
        image_dir: directory of images
        label_dir: directory of labelTxt files
        save_dir: directory to save visualization results
        show_difficult: whether to show difficult flag
        linewidth: polygon line width
        figsize: matplotlib figure size
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(image_dir)

    print(f"Found {len(image_paths)} images.")
    print(f"Image dir: {image_dir}")
    print(f"Label dir: {label_dir}")
    print(f"Save dir: {save_dir}")

    for idx, image_path in enumerate(image_paths, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"

        save_path = save_dir / f"{image_path.stem}_vis.png" if save_dir else None

        print(f"[{idx}/{len(image_paths)}] Visualizing {image_path.name}")

        try:
            visualize_dota_sample(
                image_path=image_path,
                label_path=label_path,
                save_path=save_path,
                show_difficult=show_difficult,
                linewidth=linewidth,
                figsize=figsize,
                show=show,
            )
        except FileNotFoundError as e:
            print(f"\033[31mWarning: {e}\033[0m")
            continue

    print("Done.")


if __name__ == "__main__":

    # =========================================================
    # Mode 1: visualize one image
    # =========================================================
    # image_path = r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train_lrx4\images\P0023_x0000_y4352.png"
    # label_path = r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train_lrx4\labelTxt\P0023_x0000_y4352.txt"
    # save_path = r"D:\Data\RemoteSensing\DOTA\vis\P0005_vis.png"
    #
    # visualize_dota_sample(
    #     image_path=image_path,
    #     label_path=label_path,
    #     save_path=None,
    #     show_difficult=False,
    #     show=True,
    # )

    # =========================================================
    # Mode 2: visualize all cropped patches
    # =========================================================
    image_dir = r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train_lrx4\images"
    label_dir = r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train_lrx4\labelTxt"
    save_dir = r"D:\codes\My_SR_new\DOTA_crop_dataset\dota_samples\train_lrx4\visuals"

    visualize_dota_dir(
        image_dir=image_dir,
        label_dir=label_dir,
        save_dir=save_dir,
        show_difficult=False,
        linewidth=2,
        figsize=(8, 8),
    )

    pass
