import datetime
import logging
import math
import time
import torch
from os import path as osp
import os

from simplesr.utils.options import opt_dict_to_str, parse_options
from simplesr.utils.distributed_utils import master_only

@master_only
def make_dirs(opt):
    path_opt=opt['path']
    os.makedirs(path_opt['results_root'], exist_ok=True)
    os.makedirs(path_opt['visualization'], exist_ok=True)

def test_pipeline(opt):
    opt,args=parse_options(opt)

    results_root=opt['results_root']
    resume_train=False
    resume_ckpt=opt["path"].get("resume_ckpt")
    if resume_ckpt:
        resume_train=True
    if not resume_train:
        make_dirs(opt)

    # log file path
    opt['path']['log_file'] = osp.join(results_root, 'log.txt')

