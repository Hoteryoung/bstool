import os
import csv
import numpy as np
from collections import defaultdict
import pandas
import tqdm
import shutil
import argparse

import bstool


def parse_csv(csv_file):
    csv_df = pandas.read_csv(csv_file)
    file_names = list(csv_df.ImageId.unique())

    return file_names

if __name__ == '__main__':
    version = '20201028'
    print("Processing the version of ", version)

    csv_file = f'./data/buildchange/public/{version}/shanghai_val_footprint_crop1024_gt_minarea500.csv'    
    file_names = parse_csv(csv_file)

    src_root_image_dir = './data/buildchange/v2/shanghai/images'
    src_root_label_dir = './data/buildchange/v2/shanghai/labels'
    dst_root_image_dir = './data/buildchange/public/{}/{}/images'
    dst_root_label_dir = './data/buildchange/public/{}/{}/labels'

    for file_name in tqdm.tqdm(file_names):
        base_name = bstool.get_basename(file_name)
        city = file_name.split("__")[0].split('_')[0]
        sub_fold, ori_image_name, coord = bstool.get_info_splitted_imagename(base_name)
        src_image_file = os.path.join(src_root_image_dir.format(city), base_name + '.png')
        src_label_file = os.path.join(src_root_label_dir.format(city), base_name + '.json')

        bstool.mkdir_or_exist(dst_root_image_dir.format(version, city))
        bstool.mkdir_or_exist(dst_root_label_dir.format(version, city))

        dst_image_file = os.path.join(dst_root_image_dir.format(version, city), base_name + '.png')
        dst_label_file = os.path.join(dst_root_label_dir.format(version, city), base_name + '.json')
        
        # shutil.copy(src_image_file, dst_image_file)
        # shutil.copy(src_label_file, dst_label_file)
        os.remove(dst_image_file)
        os.remove(dst_label_file)
