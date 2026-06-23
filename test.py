import logging
import os
import torch
from os import path as osp

from simplesr.data import build_dataloader, build_dataset
from simplesr.models import build_model
from simplesr.utils.distributed_utils import master_only
from simplesr.utils.log_utils import CSVLogger, get_root_logger
from simplesr.utils.misc import get_current_time
from simplesr.utils.options import opt_dict_to_str, parse_args, parse_options, save_yaml


@master_only
def make_dirs(opt):
    """创建测试阶段需要的输出目录。"""
    os.makedirs(opt['path']['results_root'], exist_ok=True)
    os.makedirs(opt['path']['visualization'], exist_ok=True)


def create_test_dataloaders(opt, logger):
    """根据配置创建测试/验证 dataloader。"""
    test_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        phase_name = phase.split('_')[0]
        if phase_name not in ['test', 'val']:
            logger.warning(f"Skip dataset phase '{phase}'. Test only uses val/test datasets.")
            continue

        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['world_size'],
            dist=opt['distributed'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(f'Number of test images/folders in {test_set.name}: {len(test_set)}')
        test_loaders.append(test_loader)

    if not test_loaders:
        raise ValueError("No val/test dataset is configured in opt['datasets'].")

    return test_loaders


def test_pipeline():
    opt, args = parse_args()
    debug = args.debug
    if debug:
        print("=" * 10 + "调试模式" + "=" * 10)

    opt = parse_options(opt, is_train=False)
    opt.setdefault('val', {})
    opt['val'].setdefault('save_img', False)

    torch.backends.cudnn.benchmark = True

    current_time = get_current_time()
    results_root = opt['path']['results_root']
    if not debug:
        make_dirs(opt)
        save_yaml(opt, osp.join(results_root, 'config.yaml'))

    opt['path']['log_file'] = osp.join(results_root, f'log_{current_time}.log')
    log_file = opt['path']['log_file'] if not debug else None
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file_path=log_file)
    logger.info("test opt:\n" + opt_dict_to_str(opt))

    csv_logger = CSVLogger(osp.join(results_root, 'test_log.csv'), flush_freq=1)

    test_loaders = create_test_dataloaders(opt, logger)

    model = build_model(opt)

    current_iter = opt['exp_name']
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.name
        logger.info(f'Testing {test_set_name}...')
        test_results = model.validation(
            test_loader,
            current_iter=current_iter,
            wandb_logger=None,
            save_img=opt['val']['save_img'])
        if test_results is not None:
            csv_logger.write({'iter': current_iter, **test_results})

    csv_logger.flush()


if __name__ == '__main__':
    test_pipeline()
