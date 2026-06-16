import argparse
import os
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp
from omegaconf import OmegaConf,DictConfig

from simplesr.utils.misc import set_random_seed,get_current_time
from simplesr.utils.distributed_utils import init_distributed_mode,master_only,is_main_process

# 解析配置

def load_yaml(yaml_file,return_dict=True):
    """从 YAML 文件路径或 YAML 字符串解析配置。
    参数:
        y (str): YAML 文件路径，或直接传入 YAML 文本字符串。
        return_dict (bool): 是否返回普通 dict。默认 True。
    返回:
        DictConfig | dict: return_dict=False 时返回 OmegaConf 配置对象，
            否则返回解析后的普通字典。
    """
    opt = OmegaConf.load(yaml_file)
    return opt if not return_dict else OmegaConf.to_container(opt,resolve=True, throw_on_missing=True)

@master_only
def save_yaml(opt: dict, file_path: str) -> None:
    """
    将字典类型的配置信息保存为 YAML 文件。

    参数:
        opt (dict): 要保存的配置字典。
        file_path (str): YAML 文件的完整保存路径。
    """
    # 将字典转换为 OmegaConf 对象
    cfg = OmegaConf.create(opt)
    # 保存到 YAML
    OmegaConf.save(config=cfg, f=file_path)


def opt_dict_to_str(opt):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    opt=OmegaConf.create(opt)
    opt_str=OmegaConf.to_yaml(opt)
    return opt_str

def overwrite_options(opt,dotlist):
    opt=load_yaml(opt,return_dict=False)
    OmegaConf.set_struct(opt, False)
    #读 CLI dotlist 覆盖（unknown 里是 ["a.b=1", "x=[1,2]"]）
    opt_cli=OmegaConf.from_dotlist(dotlist)
    opt_new=OmegaConf.merge(opt,opt_cli)
    opt_dict=OmegaConf.to_container(opt_new,resolve=True)
    return opt_dict

def parse_args(cmd_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--debug', action='store_true')

    # default True, experiment dictory with time
    parser.add_argument("--add-time", action=argparse.BooleanOptionalAction, default=True)

    # 关键：用 parse_known_args，把 dotlist 覆盖项留给 OmegaConf
    args, unknown = parser.parse_known_args(cmd_args)
    # parse yml to dict
    opt=overwrite_options(args.opt,unknown)

    # debug setting
    if args.debug and not opt['exp_name'].startswith('debug'):
        opt['exp_name'] = 'debug_' + opt['exp_name']

    return opt,args

def parse_options(opt,is_train=True):

    # distributed settings
    #             - rank: 全局rank
    #             - world_size: 总进程数
    #             - local_rank: 本地rank（GPU ID）
    #             - distributed: 是否启用分布式
    #             - gpu: 当前GPU设备ID
    init_distributed_mode(opt)

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        if opt['distributed']:
            raise ValueError(
                'In distributed training, "manual_seed" must be specified in the YAML config.'
            )
        seed = random.randint(1, 2**32 - 1)
        opt['manual_seed'] = seed

    set_random_seed(seed + opt['rank'])

    # 当前模式，影响后续模型的处理
    opt['is_train'] = is_train

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # path
    output_dir = opt['output_dir']
    current_time=get_current_time()

    opt['path']={}
    # resume
    resume_ckpt=opt["path"].get("resume_ckpt")

    if is_train:

        if resume_ckpt:
            # 真正 resume：默认使用 checkpoint 所在实验目录
            experiments_root = infer_experiment_root_from_checkpoint(resume_ckpt)
            opt['is_resume_train']=True
        else:
            # 创建新的实验目录
            experiments_root = opt['path'].get('experiments_root')
            if experiments_root is None:
                experiments_root = osp.join(output_dir, opt['exp_name'],f'train_{current_time}')

        # directory
        opt['path']['experiments_root'] = experiments_root
        opt['path']['weights'] = osp.join(experiments_root, 'weights')
        opt['path']['checkpoints'] = osp.join(experiments_root, 'checkpoints')
        #opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

        # change some options for debug mode
        if 'debug' in opt['exp_name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 10
            opt['log_settings']['print_freq'] = 1
            opt['log_settings']['save_checkpoint_freq'] = 10
            opt['train']['total_iter']=15
    else:  # test
        results_root = opt['path'].get('results_root')
        if results_root is None:
            results_root = osp.join(output_dir, opt['exp_name'],f'test_{current_time}')

        opt['path']['results_root'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt

def infer_experiment_root_from_checkpoint(checkpoint):

    ckpt_path = osp.abspath(osp.expanduser(checkpoint))
    ckpt_dir = osp.dirname(ckpt_path)
    ckpt_dir_name = osp.basename(ckpt_dir)

    if ckpt_dir_name == "checkpoints":
        return osp.dirname(ckpt_dir)

    raise ValueError(
        "Cannot infer experiment root from checkpoint path. "
        "Expected checkpoint path like: <exp_root>/checkpoints/checkpoint_xxx.pth"
    )

@master_only
def make_exp_dirs(opt):
    path_opt=opt['path'].copy()
    for key, path in path_opt.items():
            os.makedirs(path, exist_ok=True)
