
from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.object_utils import instantiate_from_config



def build_network(network_opt):
    """Build network from options.

    Args:
        network_opt (dict): Configuration for dataset. It must contain:
            path (str): Dataset path.
            kwargs (str): Dataset kwargs.
    """
    net = instantiate_from_config(network_opt)
    logger = get_root_logger()
    network_name = network_opt.get('name', '')
    logger.info(f'Network [{net.__class__.__name__}] - {network_name} is built.')
    return net
