import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from simplesr.networks import build_network
from simplesr.losses import build_loss
from simplesr.metrics import calculate_metric
from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.img_utils import tensor_to_img_array, write_rgb_float_img
from .base_model import BaseModel


class SRModel(BaseModel):
    """SimpleSR 单图像超分辨率模型。

    需要的 ``opt`` 主要字段:
        - ``is_train``: 是否为训练模式。
        - ``distributed`` / ``rank`` / ``world_size``: 分布式训练与验证控制。
        - ``network_g``: 生成器网络配置，传给 ``build_network``。
        - ``path``: 实验路径配置，常用键包括 ``weights``、``checkpoints``、
          ``visualization``、``pretrain_network_g``。
        - ``train``: 训练配置，训练模式下需要 ``pixel_opt`` 或
          ``perceptual_opt``，以及 ``optim_g``、``scheduler``、``ema_decay``。
        - ``val``: 验证配置，验证时使用 ``metrics``、``save_img``、
          ``suffix``、``pbar`` 等字段。
        - ``exp_name``: 保存验证图像和日志展示时使用的实验名。
    """

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 构建生成器网络
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # 加载预训练生成器权重
        load_path = self.opt['path'].get('pretrain_network_g', None)
        self.load_path=load_path
        if load_path is not None:
            self.load_network(self.net_g, load_path, strict=True)

        if self.is_train:
            self.init_training_settings()


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'使用 EMA，decay={self.ema_decay}')
            self.net_g_ema = self.init_ema(self.net_g, decay=self.ema_decay)

            # 若提供了预训练权重，则同步加载到 EMA 网络。
            if self.load_path is not None:
                self.load_network(self.net_g_ema.module, self.load_path, strict=True)

        self.init_losses()

        # 构建优化器和学习率调度器
        self.setup_optimizers()
        self.setup_schedulers()

    def init_losses(self):
        train_opt = self.opt["train"]

        self.cri_pix = None
        self.cri_perceptual = None

        if train_opt.get("pixel_opt") is not None:
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)

        if train_opt.get("perceptual_opt") is not None:
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(self.device)

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("At least one loss should be defined.")


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.update_ema(self.net_g, self.net_g_ema)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            net_g_ema = self.get_ema_module(self.net_g_ema)
            net_g_ema.eval()
            with torch.no_grad():
                self.output = net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # 自增强技术
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            if op == 'v':
                return torch.flip(v, dims=[3])
            elif op == 'h':
                return torch.flip(v, dims=[2])
            elif op == 't':
                return v.transpose(2, 3)
            else:
                raise ValueError(f'Unsupported self-ensemble transform: {op}')

            # 原始写法：先转 numpy 再转回 tensor。
            # 该写法会触发 GPU -> CPU -> GPU 拷贝，并可能改变 dtype，因此不再使用。
            # v2np = v.data.cpu().numpy()
            # if op == 'v':
            #     tfnp = v2np[:, :, :, ::-1].copy()
            # elif op == 'h':
            #     tfnp = v2np[:, :, ::-1, :].copy()
            # elif op == 't':
            #     tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            # ret = torch.Tensor(tfnp).to(self.device)
            # return ret

        # prepare augmented data
        lq_list = [self.lq]
        # [origin, v, h, h(v), t, t(v), t(h), t(h(v))]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            net_g_ema = self.get_ema_module(self.net_g_ema)
            net_g_ema.eval()
            with torch.no_grad():
                out_list = [net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.stack(out_list, dim=0)

        self.output = output.mean(dim=0)

    def dist_validation(self, dataloader, current_iter, wandb_logger, save_img):
        if self.opt['rank'] == 0:
            return self.nondist_validation(dataloader, current_iter, wandb_logger, save_img)
        return None

    def nondist_validation(self, dataloader, current_iter, wandb_logger, save_img):
        dataset_name = dataloader.dataset.name
        assert self.opt['val'].get('metrics') is not None
        with_metrics=True
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        num_samples = 0
        if use_pbar:
            pbar = tqdm(total=len(dataloader.dataset), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor_to_img_array(visuals['result'])
            sr_imgs = sr_img if isinstance(sr_img, list) else [sr_img]

            gt_imgs = None
            if 'gt' in visuals:
                gt_img = tensor_to_img_array(visuals['gt'])
                gt_imgs = gt_img if isinstance(gt_img, list) else [gt_img]

            lq_paths = val_data.get('lq_path', None)
            if lq_paths is None:
                lq_paths = [f'{idx:08d}_{i:04d}' for i in range(len(sr_imgs))]
            elif isinstance(lq_paths, str):
                lq_paths = [lq_paths]

            for sample_idx, sr_img in enumerate(sr_imgs):
                img_name = osp.splitext(osp.basename(lq_paths[sample_idx]))[0]

                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                    else:
                        suffix = self.opt['val'].get('suffix')
                        if suffix:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{suffix}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["exp_name"]}.png')
                    write_rgb_float_img(sr_img, save_img_path)

                if with_metrics:
                    metric_data = {'img': sr_img}
                    if gt_imgs is not None:
                        metric_data['img2'] = gt_imgs[sample_idx]
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

                num_samples += 1
                if use_pbar:
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')

            if hasattr(self, 'gt'):
                del self.gt

            # # tentative for out of GPU memory
            # del self.lq
            # del self.output
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
        if use_pbar:
            pbar.close()

        if with_metrics:
            if num_samples == 0:
                raise ValueError(f'Validation dataloader for {dataset_name} is empty.')
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= num_samples
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, wandb_logger)
            return {'dataset': dataset_name, **self.metric_results}
        return {'dataset': dataset_name}

    def _log_validation_metric_values(self, current_iter, dataset_name, wandb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if wandb_logger:
            for metric, value in self.metric_results.items():
                wandb_logger.log({f'metrics/{dataset_name}/{metric}': value}, step=current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        net_g_ema = getattr(self, 'net_g_ema', None)
        if net_g_ema is not None:
            ema_module = self.get_ema_module(net_g_ema)
            self.save_network(ema_module, 'net_g_ema', current_iter)
        self.save_checkpoint(epoch, current_iter)
