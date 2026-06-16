from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.object_utils import instantiate_from_config



def build_loss(loss_opt):
    """Build loss from options.

    Args:
        loss_opt (dict): Configuration for losses. It must contain:
            path (str): loss path.
            kwargs (str): loss kwargs.
    """
    loss = instantiate_from_config(loss_opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}].')
    return loss
