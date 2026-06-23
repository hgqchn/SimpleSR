import sys
import os
import torch
import logging
import time
import datetime
from os import path as osp

from train import create_train_val_dataloader,make_dirs

from simplesr.data import build_dataloader, build_dataset
from simplesr.data.data_sampler import EnlargedSampler
#from simplesr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from simplesr.models import build_model

from simplesr.utils.options import parse_args,parse_options,opt_dict_to_str,save_yaml
from simplesr.utils.log_utils import AvgTimer,TrainMessageLogger,get_root_logger,CSVLogger,init_wandb_logger
from simplesr.utils.distributed_utils import is_main_process,get_world_size,get_rank,master_only
from simplesr.utils.misc import get_current_time

if __name__ == '__main__':

    args_list=[
        '-opt',r"D:\codes\SimpleSR\configs\test.yaml",
        '--debug',
    ]

    pass
