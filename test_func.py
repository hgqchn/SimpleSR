import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from simplesr.utils.img_utils import img_from_bytes,img_array_to_tensor,read_img_as_rgb_float,img_rotate

if __name__ == '__main__':
    img_path = r"D:\Data\RemoteSensing\DOTA\val\images\P0003.png"
    img = read_img_as_rgb_float(img_path)
    img=img[:600,:600]
    img_rot90=np.rot90(img)

    img_rot90_v1=img_rotate(img,90)

    fig,axes=plt.subplots(1,3)
    for ax in axes:
        ax.axis('off')
    axes[0].imshow(img)
    axes[1].imshow(img_rot90)
    axes[2].imshow(img_rot90_v1)
    plt.show()
    pass
