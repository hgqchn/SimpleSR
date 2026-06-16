from copy import deepcopy
from os import path as osp

from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.object_utils import instantiate_from_config

def build_model(model_opt):
    """Build SR model from options.

    Args:
        model_opt (dict): Configuration for model. It must contain:
            path (str): Dataset path.
            kwargs (str): Dataset kwargs.
    """
    model = instantiate_from_config(model_opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model