# -*- encoding: utf-8 -*-
'''
@File    :   create_high_score_training_near_nadir.py
@Time    :   2020/12/30 22:22:05
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   根据分数高低，挑选 near-nadir 图像
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
    file_names = list(csv_df.file_name.unique())
    ori_image_names = list(csv_df.ori_image_fn.unique())
    scores = list(csv_df.score.unique())

    full_data = []

    for index in csv_df.index:
        full_data.append(csv_df.loc[index].values[0:])

    return file_names, ori_image_names, scores, full_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval on semantic segmentation')
    parser.add_argument(
        '--score_threshold',
        type=int,
        default=10000, # 17225 for 20201028 generation, 13608 for 20201027 generation, 9360 and 0 for 20201027 highset generation
        help='dataset for evaluation')
    parser.add_argument(
        '--version',
        type=str,
        default='20201119_near_nadir', 
        help='dataset for evaluation')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    version = args.version
    print("Processing the version of ", version)
    csv_file = f'./data/buildchange/high_score/misc/full_dataset_info_{version}.csv'
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    
    file_names, ori_image_names, scores, full_data = parse_csv(csv_file)

    training_info = full_data[0:3500]
    
    training_csv_file = f'./data/buildchange/high_score/misc/training_dataset_info_{version}.csv'
    
    with open(training_csv_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        head = ['file_name', 'sub_fold', 'ori_image_fn', 'coord_x', 'coord_y', 'object_num', 'mean_angle', 'mean_height', 'mean_offset_length', 'std_offset_length', 'std_angle', 'no_ignore_rate', 'score']
        csv_writer.writerow(head)
        for data in training_info:
            csv_writer.writerow(data)

    print("The number of training data: ", len(training_info))

    if len(training_info) == 3500:
        selected_image_info = []
        training_imageset_file ='./data/buildchange/high_score/misc/training_imageset_file_{}_high_score_2500_{}.txt'
        opened_file = dict()
        for city in cities:
            f = open(training_imageset_file.format(version, city), 'w')
            opened_file[city] = f

        src_root_label_dir = './data/buildchange/v2/{}/labels'
        dst_root_label_dir = './data/buildchange/high_score/{}/{}/labels'
        for data in tqdm.tqdm(training_info):
            base_name = data[0]
            city = base_name.split("__")[0].split('_')[0]
            sub_fold, ori_image_name, coord = bstool.get_info_splitted_imagename(base_name)
            src_label_file = os.path.join(src_root_label_dir.format(city), base_name + '.json')

            bstool.mkdir_or_exist(dst_root_label_dir.format(version, city))

            dst_label_file = os.path.join(dst_root_label_dir.format(version, city), base_name + '.json')
            
            shutil.copy(src_label_file, dst_label_file)

            info = f"{base_name}\n"
            if info not in selected_image_info:
                opened_file[city].write(info)
                selected_image_info.append(info)

        for city in cities:
            f = opened_file[city]
            f.close()