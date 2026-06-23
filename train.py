import datetime
import logging
import math
import time
import torch
import os
from os import path as osp

from simplesr.data import build_dataloader, build_dataset
from simplesr.data.data_sampler import EnlargedSampler
from simplesr.models import build_model

from simplesr.utils.options import parse_args,parse_options,opt_dict_to_str,save_yaml
from simplesr.utils.log_utils import AvgTimer,TrainMessageLogger,get_root_logger,CSVLogger,init_wandb_logger
from simplesr.utils.distributed_utils import master_only
from simplesr.utils.misc import get_current_time

@master_only
def make_dirs(opt):
    path_opt=opt['path']
    os.makedirs(path_opt['experiments_root'], exist_ok=True)
    os.makedirs(path_opt['weights'], exist_ok=True)
    os.makedirs(path_opt['checkpoints'], exist_ok=True)
    os.makedirs(path_opt['visualization'], exist_ok=True)


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    train_sampler = None
    total_epochs = 0
    total_iters = 0
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)

            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)

            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['world_size'],
                dist=opt['distributed'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            # 每个epoch的迭代次数，总样本数/全局batchsize
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['world_size'], dist=opt['distributed'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters

def load_checkpoint(cpt_path):
    return torch.load(cpt_path, map_location='cpu', weights_only=False)


def train_pipeline():

    opt,args=parse_args()
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

    # csv files
    csv_flush_freq = opt['log_settings'].get('csv_flush_freq', 10)
    train_csv_logger = CSVLogger(osp.join(experiments_root, 'train_log.csv'), csv_flush_freq)
    val_csv_logger = CSVLogger(osp.join(experiments_root, 'val_log.csv'), 1)

    # wandb_logger
    # initialize wandb loggers
    wandb_logger=None
    wandb_opt=opt["log_settings"].get("wandb")
    # TODO
    exp_opt={
        "exp_name": exp_name,
        "dataset_name": opt['datasets']['train'].get('name'),
        "total_iter": opt['train']['total_iter'],

    }
    if wandb_opt and not debug and wandb_opt.get('mode') != 'disabled':
        wandb_logger = init_wandb_logger(
            exp_name=exp_name,
            wandb_opt=wandb_opt,
            exp_opt=exp_opt,
            save_dir=experiments_root,
        )

    # create model
    model=build_model(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # train iter
    start_epoch = 1
    current_iter = 0
    # resume
    if resume_train:
        resume_ckpt = opt['path']['resume_ckpt']
        logger.info(f'Resuming training from checkpoint: {resume_ckpt}')
        resume_state = load_checkpoint(resume_ckpt)
        model.resume_checkpoint(resume_state)
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    # create message logger (formatted outputs)
    msg_logger = TrainMessageLogger(opt, current_iter, wandb_logger=wandb_logger)


    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    # 每个iter的数据加载耗时，每个iter的总耗时
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.perf_counter()

    for epoch in range(start_epoch,total_epochs+1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
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
            if current_iter % opt['log_settings']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                # 学习率、损失等训练结果统一放入 results，供 TrainMessageLogger 格式化。
                results = {'lrs': model.get_current_learning_rate()}
                results.update(model.get_current_log())
                log_vars['results'] = results
                msg_logger(log_vars)

                train_csv_row = {
                    'epoch': epoch,
                    'iter': current_iter,
                    'lr': model.get_current_learning_rate()[0],
                }
                train_csv_row.update(model.get_current_log())
                train_csv_logger.write(train_csv_row)

            # save checkpoint
            if current_iter % opt['log_settings']['save_checkpoint_freq'] == 0:
                logger.info('Saving checkpoint.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    val_results = model.validation(val_loader, current_iter, wandb_logger, opt['val']['save_img'])
                    if val_results is not None:
                        val_csv_logger.write({'iter': current_iter, **val_results})

            data_timer.start()
            iter_timer.start()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            val_results = model.validation(val_loader, current_iter, wandb_logger, opt['val']['save_img'])
            if val_results is not None:
                val_csv_logger.write({'iter': current_iter, **val_results})

    train_csv_logger.flush()
    val_csv_logger.flush()


if __name__ == '__main__':
    train_pipeline()
