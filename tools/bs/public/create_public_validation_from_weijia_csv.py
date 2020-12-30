# -*- encoding: utf-8 -*-
'''
@File    :   create_public_validation_from_weijia_csv.py
@Time    :   2020/12/30 22:28:37
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   根据唯嘉生成的 CSV 文件来生成验证集
'''


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

    csv_file = f'./data/buildchange/public/{version}/shanghai_val_v3_final_crop1024_footprint_gt_minarea500.csv'    
    file_names = parse_csv(csv_file)

    src_root_image_dir = './data/buildchange/v2/shanghai/images'
    src_root_label_dir = './data/buildchange/v2/shanghai/labels'
    dst_root_image_dir = './data/buildchange/public/{}/shanghai_fine/images'
    dst_root_label_dir = './data/buildchange/public/{}/shanghai_fine/labels'

    for file_name in tqdm.tqdm(file_names):
        base_name = bstool.get_basename(file_name)
        sub_fold, ori_image_name, coord = bstool.get_info_splitted_imagename(base_name)
        src_image_file = os.path.join(src_root_image_dir, base_name + '.png')
        src_label_file = os.path.join(src_root_label_dir, base_name + '.json')

        bstool.mkdir_or_exist(dst_root_image_dir.format(version))
        bstool.mkdir_or_exist(dst_root_label_dir.format(version))

        dst_image_file = os.path.join(dst_root_image_dir.format(version), base_name + '.png')
        dst_label_file = os.path.join(dst_root_label_dir.format(version), base_name + '.json')
        
        shutil.copy(src_image_file, dst_image_file)
        shutil.copy(src_label_file, dst_label_file)
        # print(src_label_file, dst_label_file)
        # os.remove(dst_image_file)
        # os.remove(dst_label_file)
