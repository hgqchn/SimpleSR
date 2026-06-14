import cv2
import lmdb
import numpy as np
import sys
import tempfile
from multiprocessing import Pool
from os import path as osp
from pathlib import Path
from tqdm import tqdm


def make_lmdb_from_imgs(data_path,
                        lmdb_path,
                        img_path_list,
                        keys,
                        batch=5000,
                        compress_level=1,
                        multiprocessing_read=False,
                        n_thread=16,
                        map_size=None):
    """根据图片创建 LMDB 数据集。

    LMDB 目录内容结构如下：

    ::

        example.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    data.mdb 和 lock.mdb 是 LMDB 标准文件，更多细节可参考：
    https://lmdb.readthedocs.io/en/release/

    meta_info.txt 是用于记录数据集元信息的文本文件。
    使用本工具准备数据集时会自动创建该文件。
    文件中每一行记录 1) 图片名（包含扩展名）、2) 图片尺寸、3) 压缩等级，
    三者之间用空格分隔。

    例如，元信息可能是：
    `000_00000000.png (720,1280,3) 1`，含义是：
    1) 图片名（包含扩展名）：000_00000000.png；
    2) 图片尺寸：(720,1280,3)；
    3) 压缩等级：1

    通常使用不带扩展名的图片名作为 LMDB 的 key。

    如果 `multiprocessing_read` 为 True，会使用多进程把所有图片先读入内存。
    因此机器需要有足够的内存。

    Args:
        data_path (str): 读取原始图片的根目录。
        lmdb_path (str): LMDB 保存路径，必须以 `.lmdb` 结尾。
        img_path_list (list[str]): 图片路径列表，通常是相对于 `data_path` 的路径。
        keys (list[str]): 写入 LMDB 时使用的 key 列表，需要与 `img_path_list` 一一对应。
        batch (int): 每处理多少张图片提交一次 LMDB 事务。默认值：5000。
        compress_level (int): 图片编码为 PNG 时使用的压缩等级。默认值：1。
        multiprocessing_read (bool): 是否使用多进程先把所有图片读入内存。
            默认值：False。
        n_thread (int): 多进程读取时使用的进程数。
        map_size (int | None): LMDB 环境的预分配空间大小。若为 None，则根据图片大小估算。
            默认值：None。
    """

    # 图片路径列表和 LMDB key 必须一一对应，否则写入后无法可靠地根据 key 找回图片。
    assert len(img_path_list) == len(keys), ('img_path_list and keys should have the same length, '
                                             f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Totoal images: {len(img_path_list)}')

    # 这里强制使用 .lmdb 后缀，避免误把普通目录当成 LMDB 输出目录。
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")

    # 已存在的 LMDB 不覆盖，防止误删或污染已经制作好的数据集。
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    if multiprocessing_read:
        # 使用多进程提前读取并编码所有图片，可以提升写入阶段速度，但会占用大量内存。
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        pbar = tqdm(total=len(img_path_list), unit='image')

        def callback(arg):
            """get the image data and update pbar."""
            # 子进程返回后，把图片字节和尺寸缓存到内存字典中。
            key, dataset[key], shapes[key] = arg
            pbar.update(1)
            pbar.set_description(f'Read {key}')

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(read_img_worker, args=(osp.join(data_path, path), key, compress_level), callback=callback)
        pool.close()
        pool.join()
        pbar.close()
        print(f'Finish reading {len(img_path_list)} images.')

    # 创建 LMDB 环境。map_size 是 LMDB 预分配的最大数据库容量。
    if map_size is None:
        # 如果未显式指定 map_size，则用第一张图的 PNG 编码大小估算总容量。
        img = cv2.imread(osp.join(data_path, img_path_list[0]), cv2.IMREAD_UNCHANGED)
        _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        data_size_per_img = img_byte.nbytes
        print('Data size per image is: ', data_size_per_img)
        data_size = data_size_per_img * len(img_path_list)
        # 预留 10 倍空间，降低因为估算偏小导致 LMDB 写入失败的概率。
        map_size = data_size * 10

    env = lmdb.open(lmdb_path, map_size=map_size)

    # 开始写入 LMDB，同时生成 meta_info.txt 记录每张图的 key、尺寸和压缩等级。
    pbar = tqdm(total=len(img_path_list), unit='chunk')
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(1)
        pbar.set_description(f'Write {key}')
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            # 多进程模式下，图片已经提前读取并编码到内存中。
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            # 普通模式下，边读取、边编码、边写入，内存占用更低。
            _, img_byte, img_shape = read_img_worker(osp.join(data_path, path), key, compress_level)
            h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # 写入元信息。这里统一记录为 .png，因为实际存入 LMDB 的是 PNG 编码字节。
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            # 分批提交事务，避免单个事务过大，也降低异常时未提交数据的规模。
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    # 提交最后一个未满 batch 的事务，并关闭文件句柄。
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')


def read_img_worker(path, key, compress_level):
    """读取单张图片并编码为 PNG 字节数组。

    Args:
        path (str): 图片路径。
        key (str): 图片对应的 LMDB key。
        compress_level (int): 图片编码为 PNG 时使用的压缩等级。

    Returns:
        tuple[str, numpy.ndarray, tuple[int, int, int]]: 返回图片 key、PNG 编码后的
            uint8 字节数组，以及图片尺寸 `(h, w, c)`。
    """

    # 保留原始通道读取图片，例如灰度图、RGB 图或带 alpha 通道的图片。
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        # 灰度图只有 h、w 两个维度，这里手动补充通道数 c=1。
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape

    # 统一编码成 PNG 字节后写入 LMDB，不直接保存原始文件字节。
    _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


class LmdbMaker():
    """逐条写入 LMDB 的工具类。

    Args:
        lmdb_path (str): LMDB 保存路径，必须以 `.lmdb` 结尾。
        map_size (int): LMDB 环境的预分配空间大小。默认值为 `1024 ** 4`，即 1TB。
        batch (int): 每写入多少张图片提交一次 LMDB 事务。默认值：5000。
        compress_level (int): 图片编码为 PNG 时使用的压缩等级。默认值：1。
    """

    def __init__(self, lmdb_path, map_size=1024**4, batch=5000, compress_level=1):
        # LmdbMaker 用于外部脚本自行准备图片字节后逐条写入 LMDB。
        if not lmdb_path.endswith('.lmdb'):
            raise ValueError("lmdb_path must end with '.lmdb'.")
        if osp.exists(lmdb_path):
            print(f'Folder {lmdb_path} already exists. Exit.')
            sys.exit(1)

        self.lmdb_path = lmdb_path
        self.batch = batch
        self.compress_level = compress_level
        # 默认 map_size 为 1TB，用于避免大数据集写入时频繁因为空间不足失败。
        self.env = lmdb.open(lmdb_path, map_size=map_size)
        self.txn = self.env.begin(write=True)
        self.txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
        self.counter = 0

    def put(self, img_byte, key, img_shape):
        # 写入一张图片的编码字节和对应元信息。
        self.counter += 1
        key_byte = key.encode('ascii')
        self.txn.put(key_byte, img_byte)
        # 记录 meta 信息，后续读取数据集时可用来恢复图片尺寸等信息。
        h, w, c = img_shape
        self.txt_file.write(f'{key}.png ({h},{w},{c}) {self.compress_level}\n')
        if self.counter % self.batch == 0:
            # 达到 batch 数量后提交一次事务，释放当前事务占用的资源。
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self):
        # 必须显式关闭，确保最后一个未满 batch 的事务也被提交。
        self.txn.commit()
        self.env.close()
        self.txt_file.close()


if __name__ == '__main__':
    test_path = r'D:\codes\SimpleSR\dataset\DOTA_crop_dataset\dota_samples\train\images'
    lmdb_path= r'D:\codes\SimpleSR\dataset\DOTA_crop_dataset\dota_samples\train.lmdb'

    from simplesr.utils.img_utils import get_image_paths
    from simplesr.utils.misc import get_filename
    img_path_list = get_image_paths(test_path)
    keys=[get_filename(p) for p in img_path_list]

    make_lmdb_from_imgs(
        test_path,
        lmdb_path,
        img_path_list,
        keys,
        compress_level=1,
        multiprocessing_read=True,
    )
    pass
