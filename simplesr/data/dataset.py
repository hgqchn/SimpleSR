import sys
import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from simplesr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from simplesr.data.transforms import augment, paired_random_crop
from simplesr.utils.img_utils import img_from_bytes,img_array_to_tensor
from simplesr.utils.color_utils import rgb2ycbcr
from simplesr.data.file_client import FileClient

class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        io_backend (dict): IO backend type and other kwargs.
        phase (str): Dataset phase, usually ``'train'``, ``'val'`` or ``'test'``.
        scale (int): Upsampling scale.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: ``'{}'``.
        gt_size (int | None): Cropped patch size for gt patches. Only used in training.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        meta_info_file (str | None): Path for meta information file.
        color (str | None): Color space flag. ``'y'`` means converting to Y channel.
        mean (list[float] | tuple[float] | None): Normalization mean.
        std (list[float] | tuple[float] | None): Normalization std.
    """

    def __init__(
        self,
        dataroot_gt,
        dataroot_lq,
        io_backend,
        phase,
        scale,
        filename_tmpl='{}',
        gt_size=None,
        use_hflip=False,
        use_rot=False,
        meta_info_file=None,
        color=None,
        mean=None,
        std=None,
    ):
        super(PairedImageDataset, self).__init__()
        self.dataroot_gt = dataroot_gt
        self.dataroot_lq = dataroot_lq
        self.io_backend_opt = io_backend.copy()
        self.phase = phase
        self.scale = scale
        self.filename_tmpl = filename_tmpl
        self.gt_size = gt_size
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.meta_info_file = meta_info_file
        self.color = color
        self.mean = mean
        self.std = std

        # file client (io backend)
        self.file_client = None

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.dataroot_lq, self.dataroot_gt]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.dataroot_lq, self.dataroot_gt], ['lq', 'gt'])
        elif self.meta_info_file is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.dataroot_lq, self.dataroot_gt],
                ['lq', 'gt'],
                self.meta_info_file,
                self.filename_tmpl,
            )
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
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.phase == 'train':
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, self.scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.use_hflip, self.use_rot)

        # color space transform
        if self.color == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.phase != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * self.scale, 0:img_lq.shape[1] * self.scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    pass
