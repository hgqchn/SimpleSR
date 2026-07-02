import datetime as dt
import logging
import time
import os
import sys
import csv
from typing import Any
from pathlib import Path

from .distributed_utils import get_rank, master_only,is_main_process

initialized_logger = {}

class AvgTimer():
    """滑动窗口平均计时器。

    用于统计训练/推理循环中每一步耗时，并提供当前步耗时与窗口平均耗时。

    参数:
        window (int): 平均窗口大小。超过该窗口后会重置累计统计，默认 200。
    """

    def __init__(self, window=200):
        self.window = window  # average window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        """开始或重置计时起点。"""
        self.start_time = self.tic = time.perf_counter()

    def record(self):
        """记录一次迭代耗时并更新平均值。"""
        self.count += 1
        self.toc = time.perf_counter()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.perf_counter()

    def get_current_time(self):
        """获取最近一次记录的耗时（秒）。"""
        return self.current_time

    def get_avg_time(self):
        """获取窗口内平均耗时（秒）。"""
        return self.avg_time




class TrainMessageLogger():
    """训练日志格式化与打印器。

    作用:
        将训练循环中的状态字典格式化成一行日志，并可选同步写入 wandb。

    参数:
        opt (dict):
            训练配置字典，当前实现至少依赖：
            - `opt['exp_name']` (str): 实验名称，用于日志展示。
            - `opt['train']['total_iter']` (int): 训练总迭代数，用于 ETA 估算。
        start_iter (int): 起始迭代步，默认 `1`。用于计算平均耗时和 ETA。
        logger (logging.Logger | None): 外部传入的日志器实例。
        wandb_logger (wandb.sdk.wandb_run.Run | None): 已初始化的 wandb 记录器；
            不为 `None` 时会把 `results` 同步写入 wandb。

    说明:
        - `reset_start_time()` 可用于重置 ETA 统计起点。
        - `__call__` 只在主进程执行（由 `@master_only` 控制）。
        - `log_dict` 中的 `results` 会先格式化拼接到文本日志，再按原字典写入 wandb。
    """

    def __init__(self, opt,start_iter=1, logger=None, wandb_logger=None):
        self.exp_name = opt['exp_name']

        self.max_iters = opt['train']['total_iter']
        self.wandb_logger = wandb_logger
        self.start_time = time.perf_counter()
        self.start_iter=start_iter
        self.logger=logger if logger else get_root_logger()

    def reset_start_time(self):
        """重置日志统计起点。"""
        self.start_time = time.perf_counter()

    @master_only
    def __call__(self, log_dict):
        """格式化并输出一条训练日志。
        参数:
            log_dict (dict):
                本次迭代的日志字段，当前实现会读取：
                - `epoch` (int): 当前 epoch。
                - `iter` (int): 当前全局迭代步。
                - `time` (float, 可选): 当前迭代耗时（秒）。
                - `data_time` (float, 可选): 当前迭代数据加载耗时（秒）。
                - `results` (dict, 可选): 训练损失、指标等其它输出字段。

        行为:
            - 先输出 `[epoch, iter]` 前缀；
            - 若存在 `time`，则追加 ETA 和 `time (data)` 信息；
            - `results` 会通过 `format_kwargs` 转成可读字符串；
            - 若配置了 `wandb_logger`，则将 `results` 以 `current_iter` 作为 step 写入 wandb。

        返回:
            None: 仅通过内部日志器输出，不返回结果。
        """
        # epoch, iter, learning rates
        epoch = log_dict.get('epoch')
        current_iter = log_dict.get('iter')

        message = (f'[epoch:{epoch:3d}, iter:{current_iter:8,d}]')

        # time and estimated time
        if 'time' in log_dict.keys():
            iter_time = log_dict.get('time')
            data_time = log_dict.get('data_time')

            total_time = time.perf_counter() - self.start_time
            # 每次迭代平均用时
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            # 估计剩余时间
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(dt.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

        # 训练损失，评估指标等其它字段
        results=log_dict.get('results')

        if self.wandb_logger is not None:
            self.wandb_logger.log(results,step=current_iter)

        results_str=format_kwargs(**results)
        message+=results_str


        self.logger.info(message)


@master_only
def init_wandb_logger(
    exp_name,
    wandb_opt,
    exp_opt,
    save_dir,
):
    """初始化 Weights & Biases 记录器。

    参数:
        exp_name (str): 实验名称，会作为 wandb run 的 `name`。
        wandb_opt (dict):
            wandb 相关配置，至少需要包含 `project`。
            支持的常用键包括：
            - `project` (str): wandb 项目名，必填。
            - `resume_id` (str | None): 断点续跑时使用的 run id。
            - `mode` (str): wandb 模式，如 `online`、`offline`、`disabled`，默认 `online`。
            - `tags` (list[str] | tuple[str] | None): run 标签列表，默认 `None`。
        exp_opt (dict):
            训练/实验的配置，会作为 `config` 传给 wandb。
        save_dir (str):
            wandb 本地保存目录，会传给 `wandb.init(..., dir=save_dir)`。

    返回:
        wandb.sdk.wandb_run.Run: 已初始化的 wandb run 对象。

    说明:
        - 如果 `resume_id` 存在，则使用该 id 并将 `resume` 设为 `allow`。
        - 如果 `resume_id` 不存在，则自动生成新的 wandb id，并将 `resume` 设为 `never`。
        - 该函数只在主进程执行。
    """
    import wandb
    logger = get_root_logger()


    name=exp_name
    wandb_opt = wandb_opt

    project = wandb_opt.get("project")
    if project is None:
        raise ValueError("['wandb']['project'] must be specified.")

    mode = wandb_opt.get("mode", "online")
    tags = wandb_opt.get("tags", None)

    # 如果 resume_id 存在，则恢复已有 run；
    # 否则生成一个新的 run id。
    resume_id = wandb_opt.get("resume_id", None)

    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb_opt["id"] = wandb_id

    run=wandb.init(id=wandb_id,
               resume=resume,
               name=name,
               config=exp_opt,
               project=project,
               tags=tags,
               mode=mode,
               dir=save_dir,
               )

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')
    return run

def get_root_logger(logger_name='basic_logger', log_level=logging.INFO, log_file_path=None):
    """获取（并按需初始化）日志器。

    行为说明:
        1. 首次调用时会创建 logger，并默认添加 ``StreamHandler``。
        2. 若 ``log_file_path`` 不为 None，且当前为 rank=0，则额外添加 ``FileHandler``。
        3. 非主进程（rank != 0）日志级别会被设为 ``ERROR``，减少重复输出。
        4. 相同 ``logger_name`` 再次调用会直接复用已有 logger。

    参数:
        logger_name (str): 日志器名称，默认 ``'basic_logger'``。
        log_level (int): 主进程日志级别，默认 ``logging.INFO``。
        log_file_path (str | None): 日志文件路径；为 None 时不写文件。

    返回:
        logging.Logger: 已初始化的日志器对象。
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    #format_str = "[%(asctime)s][%(name)s][%(filename)s][%(levelname)s] - %(message)s"

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False

    rank=get_rank()
    if rank != 0:
        logger.setLevel('ERROR')
    else:
        logger.setLevel(log_level)
    # file
    if log_file_path is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file_path, 'a')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger

def format_kwargs(**kwargs):
    """将关键字参数格式化为可读字符串。
    参数:
        **kwargs: 任意键值对，但每个值都应为单元素标量（例如 Python 数字、单元素 tensor 或单元素 numpy 标量）。

    返回:
        str: 形如 ``key1:val1  key2:val2`` 的拼接字符串，键值之间使用两个空格分隔。
    """
    str_parts=[]
    str_space=" "*2

    for key, value in kwargs.items():
        if hasattr(value, "item"):
                value = value.item()

        val_str = str(value)
        str_parts.append(f"{key}:{val_str}")
    return str_space.join(str_parts)

def _to_scalar(self,value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


class CSVLogger:
    # 保存数据到本地csv文件
    # 每轮iter的学习率，每轮iter产生的损失
    # 每次验证时的指标

    def __init__(self,csv_file,flush_freq):

        self.csv_file = Path(csv_file)
        self.buffer: list[dict[str, Any]] = []
        self.flush_freq = flush_freq
        self.fieldnames = None

    @master_only
    def write(self, row: dict[str, Any]) -> None:
        """缓存一行记录，达到 flush_freq 后写入磁盘。"""
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())

        # 简单检查：CSV 表头应保持固定。
        if list(row.keys()) != self.fieldnames:
            raise ValueError(
                "CSV row keys do not match fieldnames.\n"
                f"Expected: {self.fieldnames}\n"
                f"Got: {list(row.keys())}"
            )

        self.buffer.append(row)

        if len(self.buffer) >= self.flush_freq:
            self.flush()

    @master_only
    def flush(self) -> None:
        """把缓存中的记录写入磁盘。"""
        if not self.buffer:
            return

        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_file.exists()

        with self.csv_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.buffer)
        self.buffer.clear()

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def get_env_info():
#     """Get environment information.
#
#     Currently, only log the software version.
#     """
#     import torch
#     import torchvision
#
#     from basicsr.version import __version__
#     msg = r"""
#                 ____                _       _____  ____
#                / __ ) ____ _ _____ (_)_____/ ___/ / __ \
#               / __  |/ __ `// ___// // ___/\__ \ / /_/ /
#              / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
#             /_____/ \__,_//____//_/ \___//____//_/ |_|
#      ______                   __   __                 __      __
#     / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
#    / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
#   / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
#   \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
#     """
#     msg += ('\nVersion Information: '
#             f'\n\tBasicSR: {__version__}'
#             f'\n\tPyTorch: {torch.__version__}'
#             f'\n\tTorchVision: {torchvision.__version__}')
#     return msg
