import os
import shutil
from collections import defaultdict

import bstool


if __name__ == "__main__":
    """split ATL dataset (trainval) to train and val
    """

    src_image_dir = '/mnt/lustre/wangjinwang/data/urban3d/v1/trainval/images'
    src_label_dir = '/mnt/lustre/wangjinwang/data/urban3d/v1/trainval/labels_footprint_pixel_roof_mean'

    imageset_names_dict = defaultdict(list)
    for imageset in ['train', 'val']:
        imageset_file = f'/mnt/lustrenew/liweijia/data/urban_3d/ATL/list_trainval/region-split-v1/atl_imageid_region_{imageset}.txt'
        with open(imageset_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            imageset_names_dict[imageset].append(line.strip('\n'))

    print(imageset_names_dict)

        
    
