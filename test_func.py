import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from simplesr.utils.img_utils import img_from_bytes,img_array_to_tensor,read_img_as_rgb_float,img_rotate,get_image_paths

if __name__ == '__main__':
    img_path = r"D:\Data\RemoteSensing\DOTA\val\images"

    img_path_list=get_image_paths(img_path)
    pass
