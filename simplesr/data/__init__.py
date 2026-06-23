import importlib
import numpy as np
import random
import torch
import torch.utils.data
from copy import deepcopy
from functools import partial
from os import path as osp

from simplesr.data.prefetch_dataloader import PrefetchDataLoader
from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.object_utils import instantiate_from_config
from simplesr.utils.distributed_utils import get_rank,get_world_size


__all__ = ['build_dataset', 'build_dataloader']


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            path (str): Dataset path.
            kwargs (str): Dataset kwargs.
    """
    dataset = instantiate_from_config(dataset_opt)
    logger = get_root_logger()
    dataset_name = dataset.name
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_name} is built.')
    return dataset


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """根据数据集配置构建 PyTorch DataLoader。

    训练阶段:
        - 分布式训练时，每个进程使用 `batch_size_per_gpu` 和 `num_worker_per_gpu`。
        - 非分布式训练时，会按 `num_gpu` 放大 batch size 和 worker 数。
        - 若传入 sampler，则由 sampler 负责样本顺序，DataLoader 内部不再 shuffle。
        - 若未传入 sampler，则 DataLoader 使用 `shuffle=True` 打乱训练样本。
        - 训练默认 `drop_last=True`，丢弃最后一个不满 batch 的小批次。

    验证/测试阶段:
        - 使用 `dataset_opt.get('batch_size', 1)` 作为 batch size，未配置时默认为 1。
        - 固定 `shuffle=False`，保证评估顺序稳定。
        - 使用 `dataset_opt.get('num_workers', 0)` 作为 DataLoader worker 数，
          未配置时默认为 0；测试集较大时可配置为大于 0。

    预取模式:
        - `prefetch_mode=None`: 返回普通 DataLoader。
        - `prefetch_mode='cpu'`: 返回 PrefetchDataLoader，在后台线程预取 batch。
        - `prefetch_mode='cuda'`: 返回普通 DataLoader，后续需要配合 CUDAPrefetcher 使用。

    Args:
        dataset (torch.utils.data.Dataset): 已构建的数据集对象。
        dataset_opt (dict): 数据集配置。常用字段包括：
            phase (str): 数据阶段，支持 'train'、'val'、'test'。
            batch_size_per_gpu (int): 训练阶段每张 GPU 的 batch size。
            num_worker_per_gpu (int): 训练阶段每张 GPU 的 DataLoader worker 数。
            batch_size (int): 验证/测试阶段 batch size。默认 1。
            num_worker (int): 验证/测试阶段 DataLoader worker 数。默认 0。
            pin_memory (bool): 是否启用 DataLoader pin_memory。默认 True。
            persistent_workers (bool): 是否跨 epoch 保留 worker 进程。默认 False。
            prefetch_mode (str | None): 预取模式，可为 None、'cpu' 或 'cuda'。
            num_prefetch_queue (int): CPU 预取队列长度，仅 `prefetch_mode='cpu'` 时使用。
        num_gpu (int): GPU 数量，仅非分布式训练阶段用于放大 batch size 和 worker 数。
            默认值：1。
        dist (bool): 是否为分布式训练，仅训练阶段影响 batch size / worker 计算。
            默认值：False。
        sampler (torch.utils.data.Sampler | None): 数据采样器。分布式训练时通常传入
            DistributedSampler。默认值：None。
        seed (int | None): worker 随机种子基准值。为 None 时不设置 worker_init_fn。
            默认值：None。
    """
    phase = dataset_opt['phase']
    rank=get_rank()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier

        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        # 设置 seed 后，每个 worker 会得到不同但可复现的随机种子。
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation / testing
        batch_size = dataset_opt.get('batch_size', 1)
        num_workers = dataset_opt.get('num_worker',0)
        dataloader_args = dict(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,)
    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', True)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)

# worker 子进程初始化函数。
# 每个 worker 根据全局 seed、分布式 rank 和 worker_id 设置不同随机种子。
# seed 固定时，可保证 Python random / numpy random 相关的数据增强可复现。
def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
