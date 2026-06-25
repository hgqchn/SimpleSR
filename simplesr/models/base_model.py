import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler

from timm.utils import ModelEmaV2 as ModelEma

#from simplesr.models import lr_scheduler as lr_scheduler
from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.distributed_utils import master_only,get_local_rank


class BaseModel():
    """模型基类。"""

    def __init__(self, opt):
        self.opt = opt
        self.local_rank = get_local_rank()
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self,current_iter):
        pass

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter):
        """保存网络权重和训练状态。"""
        pass

    def validation(self, dataloader, current_iter, wandb_logger, save_img=False):
        """验证入口函数。

        参数:
            dataloader (torch.utils.data.DataLoader): 验证集 dataloader。
            current_iter (int): 当前迭代次数。
            wandb_logger: wandb 日志器。
            save_img (bool): 是否保存验证图像。默认值：False。
        """
        if self.opt['distributed']:
            return self.dist_validation(dataloader, current_iter, wandb_logger, save_img)
        else:
            return self.nondist_validation(dataloader, current_iter, wandb_logger, save_img)

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in self.opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter

    def init_ema(self,network,decay=0.999):
        net=self.get_bare_model(network)
        net_ema=ModelEma(
            net,
            decay=decay,
            device=self.device
        )
        # EMA 模型只用于滑动平均和验证，不需要梯度。
        for p in net_ema.module.parameters():
            p.requires_grad_(False)

        net_ema.module.eval()

        return net_ema

    @torch.no_grad()
    def update_ema(
        self,
        network: nn.Module | None = None,
        net_ema: ModelEma | None = None,
    ) -> None:

        net = self.get_bare_model(network)
        net_ema.update(net)

    def get_ema_module(self, net_ema):
        return net_ema.module

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net):
        """将模型移动到当前设备，并在分布式训练时封装为 DistributedDataParallel。

        参数:
            net (nn.Module): 待移动的模型。
        """
        net = net.to(self.device)
        if self.opt['distributed']:
            if self.device.type == 'cuda':
                net = DistributedDataParallel(
                    net,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False)
            else:
                net = DistributedDataParallel(net, find_unused_parameters=False)

        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_schedulers(self):
        """根据训练配置创建学习率调度器。"""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, **train_opt['scheduler']))
                #self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
                #self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        """获取未被 DistributedDataParallel 包装的原始模型。"""
        if isinstance(net, DistributedDataParallel):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """打印网络结构和参数量。

        参数:
            net (nn.Module): 待打印的网络。
        """
        if isinstance(net, DistributedDataParallel):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()
        logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        #logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """设置 warm-up 阶段的学习率。

        参数:
            lr_groups_l (list): 每个优化器对应的一组学习率。
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """获取调度器记录的初始学习率。"""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """更新学习率。

        参数:
            current_iter (int): 当前迭代次数。
            warmup_iter (int): warm-up 迭代次数。-1 表示不启用 warm-up。默认值：-1。
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # 设置 warm-up 学习率
        if current_iter < warmup_iter:
            # 获取每个参数组的初始学习率
            init_lr_g_l = self._get_init_lr()
            # 当前仅支持线性 warm-up
            warm_up_lr_l = []
            # 线性增加学习率
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # 写回学习率
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(self, net, net_label, current_iter,):
        """保存网络权重。

        参数:
            net (nn.Module): 待保存的网络。
            net_label (str): 网络标签，会用于生成文件名。
            current_iter (int): 当前迭代次数。-1 表示保存为 latest。
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['weights'], save_filename)

        net = self.get_bare_model(net)
        state_dict = net.state_dict()

        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        torch.save(state_dict, save_path)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """加载模型时打印 key 名称或张量尺寸不一致的信息。

        1. 打印当前模型和待加载权重中不一致的 key。
        2. strict=False 时，打印同名但尺寸不同的 key，并跳过这些权重。

        参数:
            crt_net (nn.Module): 当前网络。
            load_net (dict): 从文件中读取的 state_dict。
            strict (bool): 是否严格加载。默认值：True。
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # strict=False 时检查同名 key 的张量尺寸
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True):
        """加载网络权重。

        参数:
            load_path (str): 权重文件路径。
            net (nn.Module): 待加载权重的网络。
            strict (bool): 是否严格加载。
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_sd = torch.load(load_path, map_location='cpu', weights_only=False)

        logger.info(f'Loading {net.__class__.__name__} model from {load_path}')
        # # remove unnecessary 'module.'
        # for k, v in deepcopy(load_net).items():
        #     if k.startswith('module.'):
        #         load_net[k[7:]] = v
        #         load_net.pop(k)
        self._print_different_keys_loading(net, load_sd, strict)
        net.load_state_dict(load_sd, strict=strict)

    @master_only
    def save_checkpoint(self, epoch, current_iter):
        """保存训练状态，用于断点续训。

        checkpoint 保存恢复训练所需的状态：
        - epoch / iter: 当前训练进度。
        - net_g / net_g_ema: 生成器及其 EMA 权重。
        - optimizers: 所有优化器的 state_dict。
        - schedulers: 所有学习率调度器的 state_dict。

        参数:
            epoch (int): 当前 epoch。
            current_iter (int): 当前迭代次数。-1 表示 latest，此时不保存训练状态。
        """
        if current_iter != -1:
            net_g_ema = getattr(self, 'net_g_ema', None)
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'net_g': self.get_bare_model(self.net_g).state_dict(),
                'net_g_ema': net_g_ema.module.state_dict() if net_g_ema else None,
                'optimizers': [],
                'schedulers': []
            }
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['checkpoints'], save_filename)

            torch.save(state, save_path)
            return save_path
        else:
            return None


    def resume_checkpoint(self, resume_state):
        """恢复生成器、EMA、优化器和学习率调度器状态。

        参数:
            resume_state (dict): 由 checkpoint 读取到的训练状态。
        """

        if 'net_g' in resume_state:
            self.get_bare_model(self.net_g).load_state_dict(resume_state['net_g'], strict=True)

        net_g_ema = getattr(self, 'net_g_ema', None)
        if net_g_ema is not None and resume_state.get('net_g_ema') is not None:
            net_g_ema.module.load_state_dict(resume_state['net_g_ema'], strict=True)

        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """汇总 loss 字典。

        分布式训练时，会在不同 GPU 间对 loss 求平均。

        参数:
            loss_dict (OrderedDict): loss 字典。
        """
        with torch.no_grad():
            if self.opt['distributed']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
