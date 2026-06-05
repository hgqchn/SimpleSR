import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from basicsr.utils import scandir


def main():
    """多进程裁剪脚本：将大图切成小块子图，用于提升训练阶段的 IO 效率。

    这个脚本主要面向 DIV2K 这类超分训练集。
    原始图像分辨率较大，训练时如果每次都直接从整图随机裁剪，磁盘读取开销较高。
    因此这里先离线把大图切成很多固定大小的小块，后续训练直接读取这些子图。

    `opt` 中各字段含义如下：
        n_thread (int):
            并行进程数。每个进程处理一张输入图。
        compression_level (int):
            PNG 压缩等级，范围一般为 0~9。
            数值越大，文件越小，但编码更慢。这里默认 3，与 OpenCV 默认值一致。
        input_folder (str):
            待裁剪图像目录。
        save_folder (str):
            裁剪结果输出目录。
        crop_size (int):
            每个子图的边长。这里默认裁剪为正方形 patch。
        step (int):
            滑窗步长。小于 `crop_size` 时会产生重叠 patch。
        thresh_size (int):
            边界剩余区域阈值。
            如果最后剩余的宽/高超过该值，则会额外补一个贴边 patch；
            否则忽略这部分边缘区域。

    默认使用方式：
        脚本会依次处理 DIV2K 的四组目录：
        1. HR 图像
        2. X2 LR 图像
        3. X3 LR 图像
        4. X4 LR 图像

    这样做的目的，是让不同倍率下 HR/LR 子图仍然保持一一对应关系，
    便于后续组成成对训练样本。

    使用前注意：
        1. 按本地数据路径修改各目录。
        2. 如果输出目录已存在，脚本会直接退出，避免覆盖已有结果。
        3. 为了保持 HR/LR patch 数量一致，不同尺度下的 `crop_size` 和 `step`
           需要按缩放倍率成比例设置。
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    # HR 图像：以 480x480 的窗口裁剪，步长 240，产生 50% 重叠。
    opt['input_folder'] = 'D:\Data\DIV2K\DIV2K_train_HR'
    opt['save_folder'] = 'D:\Data\DIV2K\DIV2K_train_HR_sub'
    opt['crop_size'] = 480
    opt['step'] = 240
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # X2 低分辨率图像：窗口和步长都缩小为 HR 的 1/2，
    # 这样裁出来的 patch 与 HR patch 在空间位置上仍然对应。
    opt['input_folder'] = 'D:\Data\DIV2K\DIV2K_train_LR_bicubic\X2'
    opt['save_folder'] = 'D:\Data\DIV2K\DIV2K_train_LR_bicubic\X2_sub'
    opt['crop_size'] = 240
    opt['step'] = 120
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # X3 低分辨率图像：窗口和步长缩小为 HR 的 1/3。
    opt['input_folder'] = 'D:\Data\DIV2K\DIV2K_train_LR_bicubic\X3'
    opt['save_folder'] = 'D:\Data\DIV2K\DIV2K_train_LR_bicubic\X3_sub'
    opt['crop_size'] = 160
    opt['step'] = 80
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # X4 低分辨率图像：窗口和步长缩小为 HR 的 1/4。
    opt['input_folder'] = 'D:\Data\DIV2K\DIV2K_train_LR_bicubic\X4'
    opt['save_folder'] = 'D:\Data\DIV2K\DIV2K_train_LR_bicubic\X4_sub'
    opt['crop_size'] = 120
    opt['step'] = 60
    opt['thresh_size'] = 0
    extract_subimages(opt)


def extract_subimages(opt):
    """按配置批量裁剪目录中的所有图像。

    整体流程：
        1. 检查并创建输出目录。
        2. 扫描输入目录下的全部图像路径。
        3. 创建进程池。
        4. 每张图像交给一个 worker 处理。
        5. 用 tqdm 展示总体进度。

    Args:
        opt (dict): 配置字典，至少包含以下字段：
            input_folder (str): 输入目录。
            save_folder (str): 输出目录。
            n_thread (int): 并行进程数。
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        # 这里选择直接退出，而不是覆盖已有目录。
        # 原因是这个脚本通常用于一次性的离线预处理，
        # 如果重复写入同一目录，容易产生混杂结果或重复文件。
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    # 扫描输入目录中的所有文件路径。
    # `full_path=True` 表示返回绝对/完整路径，便于后续直接读取。
    img_list = list(scandir(input_folder, full_path=True))

    # 进度条统计的是“图像张数”，不是“patch 数量”。
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        # 异步提交任务：
        # - 每张大图由一个 worker 独立裁剪
        # - callback 在该图处理完成后触发，用来更新进度条
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))

    # 不再接受新任务，等待已有任务全部完成。
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """处理单张图像：按滑窗规则裁成多个子图并写盘。

    Args:
        path (str): 当前待处理图像路径。
        opt (dict): 配置字典，包含：
            crop_size (int): 裁剪窗口大小。
            step (int): 滑窗步长。
            thresh_size (int): 边界剩余区域阈值。
            save_folder (str): patch 输出目录。
            compression_level (int): PNG 压缩等级。

    Returns:
        str: 简单处理信息。当前实现中这个返回值本身没有被显示，
        但会传入 `apply_async` 的 callback。
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # 对 DIV2K 的 LR 文件名做归一化处理。
    # 例如 `0801x4.png` 会变成 `0801.png` 对应的基础名 `0801`。
    # 这样 HR 与不同尺度 LR 在裁剪后能保持统一前缀，便于后续配对。
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    # 读取原图；`IMREAD_UNCHANGED` 保留原始通道数和位深。
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]

    # 生成纵向滑窗起点。
    # 例如高度 h=1000, crop_size=480, step=240 时，先得到：
    # [0, 240, 480]
    # 表示从这些位置开始裁剪。
    h_space = np.arange(0, h - crop_size + 1, step)

    # 如果最后剩余的边界区域大于阈值，则额外补一个贴底部的 patch。
    # 这样可以尽量覆盖图像尾部，减少未被裁到的区域。
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)

    # 横向逻辑与纵向相同。
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            # NumPy 切片：从 (x, y) 开始取一个 crop_size x crop_size 的区域。
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]

            # 保证内存连续。
            # 某些 OpenCV 接口在写入非连续数组时可能触发额外复制或行为不稳定，
            # 这里显式转成连续内存布局更稳妥。
            cropped_img = np.ascontiguousarray(cropped_img)

            # patch 文件命名格式：
            # <原图名>_s001.png, <原图名>_s002.png, ...
            # 其中 `s` 可以理解为 sub-image / slice 的编号。
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])

    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    # 直接运行该文件时，按 main() 中预设的 DIV2K 配置执行。
    main()
