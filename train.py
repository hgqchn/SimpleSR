import datetime
import logging
import math
import time
import torch
from os import path as osp
import os

from simplesr.data import build_dataloader, build_dataset
from simplesr.data.data_sampler import EnlargedSampler
from simplesr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from simplesr.models import build_model


from simplesr.utils.misc import get_current_time
from simplesr.utils.log_utils import get_root_logger,init_wandb_logger,AvgTimer,TrainMessageLogger
from simplesr.utils.options import opt_dict_to_str, parse_options,save_yaml
from simplesr.utils.distributed_utils import master_only

@master_only
def make_dirs(opt):
    path_opt=opt['path']
    os.makedirs(path_opt['experimetns_root'], exist_ok=True)
    os.makedirs(path_opt['weights'], exist_ok=True)
    os.makedirs(path_opt['checkpoints'], exist_ok=True)
    #os.makedirs(path_opt['visualization'], exist_ok=True)

def train_pipeline(opt):
    current_time=get_current_time()

    opt,args=parse_options(opt)

    experiments_root=opt['experiments_root']
    resume_train=False
    resume_ckpt=opt["path"].get("resume_ckpt")
    if resume_ckpt:
        resume_train=True
    if not resume_train:
        make_dirs(opt)
        save_yaml(opt,osp.join(experiments_root,'config.yaml'))

    # log file path
    opt['path']['log_file'] = osp.join(experiments_root, f'log_{current_time}.txt')

