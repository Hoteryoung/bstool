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

        dst_image_dir = f'/mnt/lustre/wangjinwang/data/urban3d/v1/{imageset}/ATL/images'
        dst_label_dir = f'/mnt/lustre/wangjinwang/data/urban3d/v1/{imageset}/ATL/labels'
        for image_name in os.listdir(src_image_dir):
            image_basename = bstool.get_basename(image_name)
            if image_basename not in imageset_names_dict[imageset]:
                continue
            else:
                src_image_file = os.path.join(src_image_dir, image_name)
                src_label_file = os.path.join(src_label_dir, image_basename + '.json')

                dst_image_file = os.path.join(dst_image_dir, image_name)
                dst_label_file = os.path.join(dst_label_dir, image_basename + '.json')

                bstool.move_file(src_image_file, dst_image_file)
                bstool.move_file(src_label_file, dst_label_file)

        
    
