import sys
import os
from torch.utils import data
from torchvision.transforms.functional import normalize
import random

from simplesr.data.data_utils import paired_paths_from_folder, paired_paths_from_lmdb
from simplesr.utils.img_utils import img_from_bytes,img_array_to_tensor,img_augment,paired_random_crop_numpy
from simplesr.utils.color_utils import rgb2ycbcr
from simplesr.data.file_client import FileClient

class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **folder**: Scan folders to generate paths. The rest.

    Args:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        io_backend (dict): IO backend type and other kwargs.
        phase (str): Dataset phase, usually ``'train'``, ``'val'`` or ``'test'``.
        scale (int): Upsampling scale.
        lq_filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: ``'{}'``.
        gt_size (int | None): Cropped patch size for gt patches. Only used in training.
        color (str | None): Color space flag. ``'y'`` means converting to Y channel.
        mean (list[float] | tuple[float] | None): Normalization mean.
        std (list[float] | tuple[float] | None): Normalization std.
    """

    def __init__(
        self,
        dataroot_gt,
        dataroot_lq,
        phase,
        scale,
        name='',
        io_backend_opt=None,
        lq_filename_tmpl='{}', #lq_filename_template
        gt_size=None,
        use_augment=False,
        color=None,
        mean=None,
        std=None,
    ):
        super(PairedImageDataset, self).__init__()
        self.name = name
        self.dataroot_gt = dataroot_gt
        self.dataroot_lq = dataroot_lq
        self.io_backend_opt = io_backend_opt.copy()
        self.phase = phase
        self.scale = scale
        self.filename_tmpl = lq_filename_tmpl
        self.gt_size = gt_size
        self.use_augment = use_augment
        self.color = color
        self.mean = mean
        self.std = std

        # file client (io backend)
        self.file_client = None

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.dataroot_lq, self.dataroot_gt]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.dataroot_lq, self.dataroot_gt], ['lq', 'gt'])
        else:
            self.paths = paired_paths_from_folder(
                [self.dataroot_lq, self.dataroot_gt], ['lq', 'gt'], self.filename_tmpl
            )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = img_from_bytes(img_bytes)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = img_from_bytes(img_bytes)

        # augmentation for training
        if self.phase == 'train':
            if self.gt_size:
                # random crop
                img_gt, img_lq = paired_random_crop_numpy(img_gt, img_lq, self.gt_size, self.scale, gt_path)
            # flip, rotation
            if self.use_augment:
                mode=random.randint(0, 7)
                img_gt=img_augment(img_gt,mode=mode)
                img_lq=img_augment(img_lq,mode=mode)

        # color space transform
        if self.color == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt=img_array_to_tensor(img_gt)
        img_lq = img_array_to_tensor(img_lq)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    pass
