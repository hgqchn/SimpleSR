import os
import torch
from collections import OrderedDict
from torch import nn as nn
from torchvision.models import vgg as vgg

VGG_PRETRAIN_PATH=r'D:\codes\SimpleSR\pretrained_models\vgg19-dcbb9e9d.pth'
#VGG_PRETRAIN_PATH = 'experiments/pretrained_models/vgg19-dcbb9e9d.pth'

# torchvision 新版 VGG 通过 weights 参数加载预训练权重。
# 不同 VGG 构造函数对应不同的权重枚举类，例如:
#   vgg19    -> VGG19_Weights.DEFAULT
#   vgg19_bn -> VGG19_BN_Weights.DEFAULT
# 这里用字符串表建立 vgg_type 到权重枚举类名的映射，避免写一大段 if/elif。
VGG_WEIGHT_ENUMS = {
    'vgg11': 'VGG11_Weights',
    'vgg11_bn': 'VGG11_BN_Weights',
    'vgg13': 'VGG13_Weights',
    'vgg13_bn': 'VGG13_BN_Weights',
    'vgg16': 'VGG16_Weights',
    'vgg16_bn': 'VGG16_BN_Weights',
    'vgg19': 'VGG19_Weights',
    'vgg19_bn': 'VGG19_BN_Weights',
}

NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]
}


def insert_bn(names):
    """在每个卷积层后插入 BN 层名称。

    参数:
        names (list): 原始 VGG 层名称列表。

    返回:
        list: 插入 BN 层名称后的层名称列表。
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn


def get_vgg_default_weights(vgg_type):
    """获取 torchvision VGG 模型对应的默认预训练权重枚举。

    torchvision 新版本已弃用 ``pretrained=True``，推荐使用
    ``weights=VGG*_Weights.DEFAULT``。
    """
    weights_name = VGG_WEIGHT_ENUMS.get(vgg_type)
    if weights_name is None:
        raise ValueError(f'不支持的 VGG 类型: {vgg_type}')
    return getattr(vgg, weights_name).DEFAULT


def load_vgg_state_dict(vgg_net, load_path):
    """从本地路径加载 VGG 权重。"""
    state_dict = torch.load(load_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    vgg_net.load_state_dict(state_dict)


class VGGFeatureExtractor(nn.Module):
    """用于特征提取的 VGG 网络。

    该模块常用于感知损失或风格损失：输入图像后返回指定 VGG 中间层的特征。
    如果使用本地预训练权重，请确保权重文件与 ``vgg_type`` 匹配。

    参数:
        layer_name_list (list[str]): 需要返回的 VGG 层名称列表。
            示例: ['relu1_1', 'relu2_1', 'relu3_1']。
        vgg_type (str): VGG 网络类型。默认值: 'vgg19'。
        use_input_norm (bool): 是否使用 ImageNet 均值方差归一化输入。
            若为 True，输入图像应位于 [0, 1]。默认值: True。
        range_norm (bool): 是否先把 [-1, 1] 输入归一化到 [0, 1]。默认值: False。
        requires_grad (bool): 是否训练 VGG 参数。感知损失中通常为 False。默认值: False。
        remove_pooling (bool): 是否移除 VGG 中的最大池化层。默认值: False。
        pooling_stride (int): 最大池化层步长。默认值: 2。
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        # NAMES 只维护不带 BN 的基础 VGG 层名。
        # 例如 vgg19_bn 和 vgg19 的卷积、ReLU、池化结构顺序一致，
        # 但 vgg19_bn 会在每个 conv 后额外插入 BatchNorm 层。
        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            # 对 *_bn 版本，自动在 conv 层后插入 bn 层名，
            # 使层名列表能和 torchvision 的 vgg*_bn.features 一一对应。
            self.names = insert_bn(self.names)

        # 只截取会被用到的 VGG 层，避免保留无用参数。
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(weights=None)
            load_vgg_state_dict(vgg_net, VGG_PRETRAIN_PATH)
        else:
            weights = get_vgg_default_weights(vgg_type)
            vgg_net = getattr(vgg, vgg_type)(weights=weights)

        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # remove_pooling=True 时移除池化层。
                if remove_pooling:
                    continue
                else:
                    # 某些感知损失配置会修改默认池化步长。
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # ImageNet 均值，适用于范围为 [0, 1] 的 RGB 图像。
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # ImageNet 标准差，适用于范围为 [0, 1] 的 RGB 图像。
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """前向传播。

        参数:
            x (Tensor): 输入图像张量，形状为 (N, C, H, W)。

        返回:
            dict[str, Tensor]: 指定 VGG 层的特征字典。
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output

if __name__ == '__main__':
    # 使用示例：提取一张 [0, 1] RGB 图像在多个 VGG19 层上的特征。
    vgg_extractor = VGGFeatureExtractor(
        layer_name_list=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
        vgg_type='vgg19',
        use_input_norm=True,
        requires_grad=False,
    )
    img = torch.rand(1, 3, 128, 128)
    features = vgg_extractor(img)
    for name, feat in features.items():
        print(name, feat.shape)
