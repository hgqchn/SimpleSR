import functools
import os
import torch
import torch.distributed as dist
from typing import Any



def is_dist_avail_and_initialized():
    """检查分布式是否可用且已初始化"""
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    """获取总进程数（GPU数量）"""
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank():
    """获取当前进程的全局rank"""
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_local_rank():
    """获取当前进程的本地rank（单机内的GPU ID）"""
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process():
    """判断是否为主进程（rank 0）"""
    return get_rank() == 0

def setup_for_distributed(is_master):
    """
    禁用非主进程的打印，避免日志重复
    调用后，只有主进程会打印信息
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(cfg):
    """
    初始化分布式训练环境

    参数:
        cfg: 配置字典，会被添加以下字段：
            - rank: 全局rank
            - world_size: 总进程数
            - local_rank: 本地rank（GPU ID）
            - distributed: 是否启用分布式
            - gpu: 当前GPU设备ID
    """
    # torchrun 会自动提供这些环境变量
    # 每张GPU一个进程，torchrun会为所有进程设置环境变量
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # 全局GPU的编号，全局指的是多个节点/机器上的所有GPU
        # world_size 是所有节点/机器上的GPU总数
        cfg["rank"] = int(os.environ["RANK"])
        cfg["world_size"] = int(os.environ["WORLD_SIZE"])
        cfg["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))
        cfg["gpu"] = cfg["local_rank"]
        cfg["distributed"] = cfg["world_size"] > 1
    else:
        print("未使用分布式模式")
        cfg["rank"] = 0
        cfg["world_size"] = 1
        cfg["local_rank"] = 0
        cfg["gpu"] = 0
        cfg["distributed"] = False

    if not cfg["distributed"]:
        print('分布式未初始化，仅使用单进程')
        return

    # 每个进程绑定到对应的GPU
    torch.cuda.set_device(cfg["local_rank"])

    # 进程后端
    # nccl是NVIDIA GPU最快的后端
    backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"

    print(f"| 初始化分布式 (rank {cfg['rank']}, local_rank {cfg['local_rank']}): env://", flush=True)

    # 初始化进程组
    # env:// 表示：通信需要的地址/端口等信息由环境变量提供（torchrun 会设置 MASTER_ADDR, MASTER_PORT）
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=cfg["world_size"],
        rank=cfg["rank"]
    )

    # 进程间同步
    dist.barrier()

    # 设置只有主进程打印
    setup_for_distributed(cfg["rank"] == 0)

    # 注意：这里的 print 会被 setup_for_distributed 重定向，
    # 所以只有主进程会打印，或者当 force=True 时打印
    print("分布式训练初始化成功！")
    print(f"总进程数: {cfg['world_size']}, 全局 rank: {cfg['rank']}, 本地 rank: {cfg['local_rank']}")

def cleanup_distributed():
    """清理分布式训练环境"""
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def reduce_dict(
    input_dict: dict[str, torch.Tensor | float | int],
    average: bool = True,
    to_float: bool = True,
) -> dict[str, torch.Tensor | float]:
    """
    对所有进程上的指标字典进行 all_reduce 同步。

    常用于 DDP 训练时同步 loss / metric，避免每张 GPU 记录不同的数值。

    Args:
        input_dict:
            待同步的字典。value 可以是 Tensor、float 或 int。
            推荐传入标量，例如：
                {
                    "loss": loss.detach(),
                    "pixel_loss": pixel_loss.detach(),
                    "diffusion_loss": diffusion_loss.detach(),
                }

        average:
            是否对所有进程的结果求平均。
            - True: 返回所有进程的平均值，最常用。
            - False: 返回所有进程的求和值。

        to_float:
            是否把返回值从 Tensor 转成 Python float。
            - True: 适合日志记录、wandb、tensorboard。
            - False: 保留 Tensor，适合后续继续做 Tensor 运算。

    Returns:
        reduced_dict:
            同步后的字典。key 与 input_dict 一致。

    Example:
        >>> # 假设有 4 张 GPU：
        >>> # GPU0 loss = 0.5
        >>> # GPU1 loss = 0.3
        >>> # GPU2 loss = 0.4
        >>> # GPU3 loss = 0.6
        >>>
        >>> log_dict = {
        ...     "loss": loss.detach(),
        ...     "pixel_loss": pixel_loss.detach(),
        ... }
        >>> log_dict = reduce_dict(log_dict, average=True, to_float=True)
        >>>
        >>> # 所有 GPU 上得到相同结果：
        >>> # log_dict["loss"] = 0.45
        >>> # 然后只在主进程写日志：
        >>> if is_main_process():
        ...     logger.info(log_dict)
        ...     wandb.log(log_dict, step=current_iter)

    Notes:
        1. 所有进程必须以相同的 key 调用 reduce_dict。
           否则可能导致通信不一致或程序卡住。

        2. 函数内部会对 key 排序，保证不同进程的 reduce 顺序一致。

        3. 这个函数适合同步标量，不适合同步大 tensor。
           大 tensor 请使用 all_gather_tensor。
    """
    world_size = get_world_size()
    # 非分布式或单卡情况
    if world_size < 2:
        if to_float:
            return {
                k: float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
                for k, v in input_dict.items()
            }
        return {
            k: v if torch.is_tensor(v) else torch.tensor(v)
            for k, v in input_dict.items()
        }

    if len(input_dict) == 0:
        return {}

    with torch.no_grad():
        names = sorted(input_dict.keys())

        values = []
        device = None

        # 优先使用已有 Tensor 的 device。
        for name in names:
            value = input_dict[name]
            if torch.is_tensor(value):
                device = value.device
                break

        # 如果字典里全是 float/int，则默认放到当前 cuda 或 cpu。
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        for name in names:
            value = input_dict[name]

            if not torch.is_tensor(value):
                value = torch.tensor(value, dtype=torch.float32, device=device)
            else:
                value = value.detach()
                if value.ndim != 0:
                    value = value.mean()
                value = value.to(device=device, dtype=torch.float32)

            values.append(value)

        values = torch.stack(values, dim=0)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)

        if average:
            values /= world_size

        if to_float:
            reduced_dict = {
                name: float(value.detach().cpu().item())
                for name, value in zip(names, values)
            }
        else:
            reduced_dict = {
                name: value
                for name, value in zip(names, values)
            }

    return reduced_dict


def all_gather_tensor(data: torch.Tensor) -> list[torch.Tensor]:
    """
    收集所有进程上的 Tensor。

    Args:
        data:
            当前进程上的 Tensor。
            要求所有进程上的 Tensor shape 一致。

    Returns:
        data_list:
            长度为 world_size 的 list。
            data_list[i] 是 rank i 上的 Tensor。

    Example:
        >>> # 每张 GPU 上有一部分预测结果：
        >>> # rank0: pred shape [8, 1, H, W]
        >>> # rank1: pred shape [8, 1, H, W]
        >>>
        >>> pred_list = all_gather_tensor(pred)
        >>> pred_all = torch.cat(pred_list, dim=0)
        >>>
        >>> # pred_all shape = [world_size * 8, 1, H, W]
        >>> # 可以用于统一计算验证指标。

    Notes:
        1. 所有进程上的 data shape 必须一致。
        2. 如果最后一个 batch 每个进程大小不同，建议：
           - 验证时使用 DistributedSampler(drop_last=False) 并小心处理；
           - 或者改用 all_gather_object 收集变长结果；
           - 或者在 metric 内部先局部统计，再用 reduce_dict 汇总标量。
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    data_list = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(data_list, data)
    return data_list


def all_gather_object(data: Any) -> list[Any]:
    """
    收集所有进程上的 Python 对象。

    Args:
        data:
            任意可 pickle 的 Python 对象。
            例如 dict、list、str、float、文件名列表、metric 结果等。

    Returns:
        object_list:
            长度为 world_size 的 list。
            object_list[i] 是 rank i 上的对象。

    Example:
        >>> # 每张 GPU 验证自己的一部分图像，并得到局部结果：
        >>> local_results = [
        ...     {"name": "img_001.tif", "psnr": 31.2},
        ...     {"name": "img_002.tif", "psnr": 30.8},
        ... ]
        >>>
        >>> gathered_results = all_gather_object(local_results)
        >>>
        >>> if is_main_process():
        ...     all_results = []
        ...     for part in gathered_results:
        ...         all_results.extend(part)
        ...     mean_psnr = sum(x["psnr"] for x in all_results) / len(all_results)

    Notes:
        1. all_gather_object 比 all_gather_tensor 更灵活，但通常更慢。
        2. 适合收集文件名、metric dict、不同长度的 list。
        3. 不建议用它频繁收集大数组或大 Tensor。
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    object_list = [None for _ in range(world_size)]
    dist.all_gather_object(object_list, data)
    return object_list

def save_weight_on_master(*args, **kwargs):
    """只在主进程保存模型"""
    if is_main_process():
        torch.save(*args, **kwargs)

def master_only(func):
    """仅在主进程（rank=0）执行被装饰函数。

    典型用途：
        1. 只在主进程打印日志，避免多卡重复输出。
        2. 只在主进程保存模型/写文件，避免并发写冲突。

    说明：
        非主进程调用该函数时会直接返回 `None`。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper