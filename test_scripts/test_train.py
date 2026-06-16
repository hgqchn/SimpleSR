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

    opt,args=parse_args(args_list)
    debug=args.debug
    if debug:
        print("="*10+"调试模式"+"="*10)

    # parse options, set distributed setting, set random seed
    opt = parse_options(opt,is_train=True)

    exp_name=opt['exp_name']
    resume_train=opt.get('is_resume_train',False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    current_time=get_current_time()

    # 创建目录
    experiments_root=opt['path']['experiments_root']
    if not resume_train and not debug:
        make_dirs(opt)
        save_yaml(opt,osp.join(experiments_root,'config.yaml'))

    # log file path
    opt['path']['log_file'] = osp.join(experiments_root, f'log_{current_time}.log')
    log_file=opt['path']['log_file'] if not debug else None
    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file_path=log_file)
    logger.info("experiment opt:\n"+opt_dict_to_str(opt))

    # wandb_logger
    # initialize wandb loggers
    wanbdb_logger=None
    wandb_opt=opt["log_settings"].get("wandb")
    exp_opt={
        "exp_name": exp_name,
        "dataset_name": opt['datasets']['train'].get('name'),
        "total_iter": opt['train']['total_iter'],

    }
    if wandb_opt and not debug:
        wandb_logger = init_wandb_logger(
            exp_name=exp_name,
            wandb_opt=wandb_opt,
            exp_opt=exp_opt,
            save_dir=experiments_root,
        )

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result


    # create model
    model=build_model(opt)

    # train iter
    start_epoch = 1
    current_iter = 0
    # resume
    if resume_train:
        pass

    # create message logger (formatted outputs)
    msg_logger = TrainMessageLogger(opt, current_iter)

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    # 每个iter的数据加载耗时，每个iter的总耗时
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.perf_counter()

    for epoch in range(start_epoch,total_epochs+1):
        for train_data in train_loader:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break

            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                # 损失
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter,, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, , opt['val']['save_img'])

    pass
