import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F


from simplesr.utils.img_utils import img_array_to_tensor,get_image_paths,crop_img_by_scale

def read_img_seq(path, require_mod_crop=False, scale=1, return_imgname=False):
    """从给定文件夹路径读取一组图像序列。

    参数:
        path (list[str] | str): 图像路径列表或图像文件夹路径。
        require_mod_crop (bool): 是否对每张图像执行按比例裁剪。
            默认值: False。
        scale (int): mod_crop 的缩放因子。默认值: 1。
        return_imgname(bool): 是否返回图像名称。默认值: False。

    返回:
        Tensor: 尺寸为 (t, c, h, w)，RGB，取值范围 [0, 1]。
        list[str]: 返回的图像名称列表。
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = get_image_paths(path)
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]

    if require_mod_crop:
        imgs = [crop_img_by_scale(img, scale) for img in imgs]
    imgs = img_array_to_tensor(imgs, return_batch_tensor=True)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs

# for video
def generate_frame_indices(crt_idx, max_frame_num, num_frames, padding='reflection'):
    """生成用于从图像序列中读取 `num_frames` 帧的索引列表。

    参数:
        crt_idx (int): 当前中心索引。
        max_frame_num (int): 图像序列的最大帧数（从 1 开始）。
        num_frames (int): 要读取的帧数。
        padding (str): 填充模式，可选值为
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            示例: current_idx = 0, num_frames = 5
            不同填充模式下生成的帧索引:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    返回:
        list[int]: 索引列表。
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # 从 0 开始
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_lmdb(folders, keys):
    """从 lmdb 文件生成配对路径。

    lmdb 的内容。以 `lq.lmdb` 为例，文件结构如下:

    ::

        lq.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    data.mdb 和 lock.mdb 是标准 lmdb 文件，更多细节可参考
    https://lmdb.readthedocs.io/en/release/。

    meta_info.txt 是用于记录数据集元信息的指定 txt 文件。使用我们提供的
    数据集工具准备数据集时会自动创建该文件。
    txt 文件中的每一行记录:
    1)图像名称（含扩展名），
    2)图像形状，
    3)压缩等级，各字段之间用空格分隔。
    示例: `baboon.png (120,125,3) 1`

    我们使用不含扩展名的图像名称作为 lmdb key。
    注意，对应的 lq 和 gt 图像使用相同的 key。

    参数:
        folders (list[str]): 文件夹路径列表。列表顺序应为
            [input_folder, gt_folder]。
        keys (list[str]): 用于标识文件夹的 key 列表。顺序应与 folders
            保持一致，例如 ['lq', 'gt']。
            注意，此 key 与 lmdb key 不同。

    返回:
        list[str]: 返回的路径列表。
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(f'{input_key} folder and {gt_key} folder should both in lmdb '
                         f'formats. But received {input_key}: {input_folder}; '
                         f'{gt_key}: {gt_folder}')
    # 确保两个 meta_info 文件相同
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(dict([(f'{input_key}_path', lmdb_key), (f'{gt_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_folder(folders, keys, input_filename_tmpl='{}'):
    """从文件夹生成配对路径。

    参数:
        folders (list[str]): 文件夹路径列表。列表顺序应为
            [input_folder, gt_folder]。
        keys (list[str]): 用于标识文件夹的 key 列表。顺序应与 folders
            保持一致，例如 ['lq', 'gt']。
        input_filename_tmpl (str): 每个文件名的模板。注意该模板不包含
            文件扩展名。通常 filename_tmpl 用于输入文件夹中的文件。

    返回:
        list[str]: 返回的路径列表。
         {f'{input_key}_path': input_path, f'{gt_key}_path': gt_path}
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = get_image_paths(input_folder)
    gt_paths = get_image_paths(gt_folder)
    assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_name = f'{input_filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        assert input_path in input_paths, f'{input_name} is not in {input_key}_paths.'
        gt_path = osp.join(gt_folder, gt_path)
        paths.append({
                    f'{input_key}_path': input_path,
                    f'{gt_key}_path': gt_path,
                })
    return paths



def paths_from_lmdb(folder):
    """从 lmdb 生成路径。

    参数:
        folder (str): 文件夹路径。

    返回:
        list[str]: 返回的路径列表。
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """生成 `duf_downsample` 中使用的高斯核。

    参数:
        kernel_size (int): 核大小。默认值: 13。
        sigma (float): 高斯核的 sigma。默认值: 1.6。

    返回:
        np.array: 高斯核。
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # 将中心元素设为 1，作为狄拉克 delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # 对该 delta 进行高斯平滑，得到高斯滤波器
    return filters.gaussian_filter(kernel, sigma)


# for video
def duf_downsample(x, kernel_size=13, scale=4):
    """使用 DUF 官方代码中的高斯核进行下采样。

    参数:
        x (Tensor): 待下采样的帧，形状为 (b, t, c, h, w)。
        kernel_size (int): 核大小。默认值: 13。
        scale (int): 下采样因子。支持的 scale: (2, 3, 4)。
            默认值: 4。

    返回:
        Tensor: DUF 下采样后的帧。
    """
    assert scale in (2, 3, 4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x
