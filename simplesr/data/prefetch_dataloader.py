import queue as Queue
import threading
import torch
from torch.utils.data import DataLoader


# 数据预取模式说明：
#
# prefetch_mode 为 None 时，使用普通 PyTorch DataLoader。
# CPUPrefetcher 只是统一训练循环接口，提供 reset() 和 next()，
# 本身不额外做预取。
#
# prefetch_mode == 'cpu' 时，使用 PrefetchDataLoader。
# PrefetchDataLoader 会用后台线程提前从 DataLoader 取 batch，
# 并放入一个有限长度的队列中。真正的 CPU 预取发生在
# PrefetchDataLoader + PrefetchGenerator 中。
#
# prefetch_mode == 'cuda' 时，使用普通 DataLoader + CUDAPrefetcher。
# CUDAPrefetcher 会提前取下一批数据，并用独立 CUDA stream
# 将 batch 字典中的 tensor 异步搬到 GPU。该模式通常需要
# pin_memory=True，并且会额外占用 GPU 显存。
#
# BasicSR 训练循环统一使用：
#     prefetcher.reset()
#     data = prefetcher.next()
#     while data is not None:
#         ...
#         data = prefetcher.next()
#
# 如果只使用普通 DataLoader，也可以不使用 prefetcher，直接写：
#     for data in dataloader:
#         ...


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Reference: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        # 设置为守护线程
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """Prefetch version of dataloader.

    Reference: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    TODO:
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher():
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher():
    """CUDA prefetcher.

    Reference: https://github.com/NVIDIA/apex/issues/304#

    It may consume more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()
