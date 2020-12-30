# -*- encoding: utf-8 -*-
'''
@File    :   image_num_height_angle_2048_v1.py
@Time    :   2020/12/30 22:24:24
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   按照图像分数对 2048 * 2048 的图像进行排序
'''


import os
import numpy as np
import glob
import tqdm
import math
import argparse
from collections import defaultdict
import csv

import bstool


class CountImage():
    def __init__(self,
                 core_dataset_name='buildchange',
                 version='v0',
                 city='shanghai',
                 sub_fold='arg',
                 resolution=0.6):
        self.city = city
        self.sub_fold = sub_fold
        self.resolution = resolution

        self.image_dir = f'./data/{core_dataset_name}/{version}/{city}/{sub_fold}/images'
        if city == 'chengdu':
            self.json_dir = f'/mnt/lustre/menglingxuan/buildingwolf/20200329/{city}_L18/{sub_fold}/anno_20200924/OffsetField/TXT'
        else:
            self.json_dir = f'/mnt/lustre/menglingxuan/buildingwolf/20200329/{city}_18/{sub_fold}/anno_20200924/OffsetField/TXT'

    def count_image(self, json_file):
        file_name = bstool.get_basename(json_file)
        objects = bstool.lingxuan_json_parser(json_file)

        # 2. skip the empty image
        if len(objects) == 0:
            return

        origin_properties = [obj['property'] for obj in objects]

        # 3. obtain the info of building
        heights = np.array([3 * obj['Floor'] for obj in origin_properties])
        offsets = np.array([[obj['xoffset'], obj['yoffset']] for obj in origin_properties])
        ignores = np.array([obj['ignore'] for obj in origin_properties])

        # 4. drop ignored objects
        keep_inds = (ignores == 0)
        heights, offsets = heights[keep_inds], offsets[keep_inds]

        # 5. get angles and offset length
        angles, offset_lengths = [], []
        for height, offset in zip(heights, offsets):
            offset_x, offset_y = offset

            angle = math.atan2(math.sqrt(offset_x ** 2 + offset_y ** 2) * self.resolution, height)
        
            offset_length = math.sqrt(offset_x ** 2 + offset_y ** 2)

            angles.append(angle)
            offset_lengths.append(offset_length)

        # 6. judge whether or not keep this image
        image_info = self.get_image_info(file_name, angles, heights, offset_lengths, ignores)
        
        return image_info

    def get_image_info(self, file_name, angles, heights, offset_lengths, ignores):
        if data_source == 'local':
            parameters = {
                        'object_num': 0,
                        "mean_height": 0,
                        "mean_angle": 0, 
                        "mean_offset_length": 0,
                        'std_offset_length': 50,
                        'std_angle': 100,
                        'no_ignore_rate': 0.9}
        else:
            parameters = {
                        'object_num': 5,
                        "mean_height": 3,
                        "mean_angle": 40, 
                        "mean_offset_length": 5,
                        'std_offset_length': 5,
                        'std_angle': 30,
                        'no_ignore_rate': 0.80}
        angles = np.abs(angles) * 180.0 / math.pi
        offset_lengths = np.abs(offset_lengths)

        mean_angle = np.mean(angles)
        mean_height = np.mean(heights)
        mean_offset_length = np.mean(offset_lengths)

        std_offset_length = np.std(offset_lengths)
        std_angle = np.std(angles)

        ignores = ignores.tolist()
        no_ignore_rate = ignores.count(0) / len(ignores)
        object_num = len(ignores)

        if object_num < parameters['object_num'] or no_ignore_rate < parameters['no_ignore_rate'] or mean_height < parameters['mean_height'] or std_angle > parameters['std_angle']:
            return

        if mean_angle < parameters['mean_angle'] and mean_offset_length < parameters['mean_offset_length'] and std_offset_length < parameters['std_offset_length']:
            return

        score = (object_num / 65) * (mean_angle / 90) * (mean_height / 10) * (mean_offset_length / 20) * (std_offset_length / 10) * (20 / (std_angle + 1)) * (no_ignore_rate)

        image_info = [file_name, object_num, mean_angle, mean_height, mean_offset_length, std_offset_length, std_angle, no_ignore_rate, score]

        return image_info

    def core(self):
        json_file_list = glob.glob("{}/*.json".format(self.json_dir))

        image_infos = []
        for json_file in tqdm.tqdm(json_file_list):
            image_info = self.count_image(json_file)
            
            if image_info is not None:
                image_infos.append(image_info)
            else:
                continue
        
        return image_infos
        
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval on semantic segmentation')
    parser.add_argument(
        '--source',
        type=str,
        default='local', 
        help='dataset for evaluation')

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()

    core_dataset_name = 'buildchange'
    version = 'v0'

    data_source = args.source   # remote or local

    if data_source == 'local':
        cities = ['shanghai']
        sub_folds = {'shanghai': ['arg']}
    else:
        cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
        sub_folds = {'beijing':  ['arg', 'google', 'ms', 'tdt'],
                    'chengdu':  ['arg', 'google', 'ms', 'tdt'],
                    'haerbin':  ['arg', 'google', 'ms'],
                    'jinan':    ['arg', 'google', 'ms', 'tdt'],
                    'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}
    
    training_image_info = []
    for city in cities:
        for sub_fold in sub_folds[city]:
            print("Begin processing {} {} set.".format(city, sub_fold))
            count_image = CountImage(core_dataset_name=core_dataset_name,
                                    version=version,
                                    city=city,
                                    sub_fold=sub_fold)
            image_infos = count_image.core()
            training_image_info.extend(image_infos)

            print("Finish processing {} {} set.".format(city, sub_fold))

    full_csv_file = f'./data/buildchange/high_score/misc/full_dataset_info_20201106.csv'

    training_image_info_ = np.array(training_image_info)
    scores = training_image_info_[:, -1].astype(np.float64)
    sorted_index = np.argsort(scores)[::-1]
    
    training_image_info = [training_image_info[idx] for idx in sorted_index]

    with open(full_csv_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        head = ['ori_image_fn', 'object_num', 'mean_angle', 'mean_height', 'mean_offset_length', 'std_offset_length', 'std_angle', 'no_ignore_rate', 'score']
        csv_writer.writerow(head)
        for data in training_image_info:
            csv_writer.writerow(data)