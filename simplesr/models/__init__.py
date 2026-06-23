from copy import deepcopy
from os import path as osp

from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.object_utils import instantiate_from_config

def build_model(opt):
    """根据完整实验配置构建模型。

    参数:
        opt (dict): 完整实验配置。必须包含 ``model`` 字段：
            - ``opt['model']['path']``: 模型类的导入路径。
            - ``opt['model']['kwargs']``: 传给模型类的额外参数，可省略。

    说明:
        当前项目的模型类通常需要完整 ``opt``，例如网络结构、训练损失、
        优化器、路径、验证配置等。因此这里会把完整 ``opt`` 作为关键字参数
        传给模型类构造函数。
    """
    model_opt = opt['model']
    model = instantiate_from_config(model_opt, extra_kwargs={'opt': opt})
    name = model_opt.get('name', '-')
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__} - {name}] is created.')
    return model
