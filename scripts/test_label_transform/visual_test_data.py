import sys
import os
from DOTA_utils.dota_visual import visualize_dota_dir
if __name__ == '__main__':
    imgs_path=r'D:\codes\SimpleSR\scripts\test_label_transform\test_data\imgs'
    labels_path=r'D:\codes\SimpleSR\scripts\test_label_transform\test_data\labels'

    visual_path=r'D:\codes\SimpleSR\scripts\test_label_transform\test_data\visuals'
    visualize_dota_dir(imgs_path,labels_path,save_dir=visual_path)

    pass
