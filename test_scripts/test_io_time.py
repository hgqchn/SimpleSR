import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simplesr.utils.img_utils import read_img_as_rgb_float,img_from_bytes,get_image_paths
from simplesr.data.file_client import FileClient
from simplesr.utils.log_utils import AvgTimer


def benchmark_read_img_as_rgb_float(img_paths, repeat=1, warmup=3):
    """测试直接从路径读取图像并转为 numpy 数组的耗时。"""
    for img_path in img_paths[:warmup]:
        read_img_as_rgb_float(img_path)

    timer = AvgTimer(window=len(img_paths) * repeat + 1)
    shape_checksum = 0
    for _ in range(repeat):
        for img_path in img_paths:
            timer.start()
            img = read_img_as_rgb_float(img_path)
            timer.record()
            shape_checksum += img.shape[0] + img.shape[1]

    return timer.get_avg_time(), shape_checksum


def benchmark_file_client_bytes_to_numpy(img_paths, repeat=1, warmup=3):
    """测试先读取图像字节，再解码为 numpy 数组的耗时。"""
    file_client = FileClient(backend='disk')

    for img_path in img_paths[:warmup]:
        img_from_bytes(file_client.get(img_path))

    timer = AvgTimer(window=len(img_paths) * repeat + 1)
    shape_checksum = 0
    for _ in range(repeat):
        for img_path in img_paths:
            timer.start()
            img_bytes = file_client.get(img_path)
            img = img_from_bytes(img_bytes)
            timer.record()
            shape_checksum += img.shape[0] + img.shape[1]

    return timer.get_avg_time(), shape_checksum


if __name__ == '__main__':

    test_dir=r'D:\codes\SimpleSR\dataset\DOTA_crop_dataset\dota_samples\train\images'
    repeat = 5
    warmup = 10
    max_images = None

    img_paths = get_image_paths(test_dir)
    if max_images is not None:
        img_paths = img_paths[:max_images]

    if not img_paths:
        raise FileNotFoundError(f'测试文件夹中没有找到图像: {test_dir}')

    print(f'测试目录: {test_dir}')
    print(f'图像数量: {len(img_paths)}')
    print(f'重复次数: {repeat}')
    print()

    bytes_avg_time, bytes_checksum = benchmark_file_client_bytes_to_numpy(
        img_paths, repeat=repeat, warmup=warmup)
    direct_avg_time, direct_checksum = benchmark_read_img_as_rgb_float(
        img_paths, repeat=repeat, warmup=warmup)


    if direct_checksum != bytes_checksum:
        raise RuntimeError(
            f'两种读取方式得到的图像尺寸校验不一致: '
            f'direct={direct_checksum}, bytes={bytes_checksum}')

    direct_total_time = direct_avg_time * len(img_paths) * repeat
    bytes_total_time = bytes_avg_time * len(img_paths) * repeat

    print('测试结果:')
    print(f'  直接读取为 numpy: 平均 {direct_avg_time * 1000:.4f} ms/张, 总耗时 {direct_total_time:.4f} s')
    print(f'  读取字节再转 numpy: 平均 {bytes_avg_time * 1000:.4f} ms/张, 总耗时 {bytes_total_time:.4f} s')

    if direct_avg_time < bytes_avg_time:
        ratio = bytes_avg_time / direct_avg_time
        print(f'结论: 直接读取为 numpy 更快，约快 {ratio:.2f}x。')
    elif bytes_avg_time < direct_avg_time:
        ratio = direct_avg_time / bytes_avg_time
        print(f'结论: 读取字节再转 numpy 更快，约快 {ratio:.2f}x。')
    else:
        print('结论: 两种方式耗时基本一致。')
