import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas
import cv2
import glob
from multiprocessing import Pool
from functools import partial
import tqdm
import math
import shutil
import argparse
from collections import defaultdict

import bstool


class CountImage():
    def __init__(self,
                 core_dataset_name='buildchange',
                 version='v2',
                 city='shanghai',
                 sub_fold='arg',
                 resolution=0.6,
                 height_angle_save_dir=None,
                 val_image_info_dir=None):
        self.city = city
        self.sub_fold = sub_fold
        self.resolution = resolution

        self.image_dir = f'./data/{core_dataset_name}/{version}/{city}/images'
        if data_source == 'local':
            self.json_dir = f'./data/{core_dataset_name}/{version}/{city}/labels'
        else:
            self.json_dir = f'./data/{core_dataset_name}/{version}/{city}/labels'

        self.height_angle_save_dir = height_angle_save_dir

        self.val_image_list = self.get_val_image_list(val_image_info_dir)

    def get_val_image_list(self, val_image_info_dir):
        if val_image_info_dir is None:
            return []

        val_image_list = []
        for sub_fold in os.listdir(val_image_info_dir):
            sub_dir = os.path.join(val_image_info_dir, sub_fold)
            for file_name in os.listdir(sub_dir):
                val_image_list.append(bstool.get_basename(file_name))

        val_image_list = list(set(val_image_list))

        return val_image_list

    def count_image(self, json_file):
        file_name = bstool.get_basename(json_file)
        sub_fold, ori_image_fn, coord = bstool.get_info_splitted_imagename(file_name)

        # 0. keep the specific sub fold image
        if sub_fold != self.sub_fold:
            return
        
        # 1. skip the validation image
        if file_name in self.val_image_list:
            print(f"This image is in val list: {file_name}")
            return
        
        objects = bstool.bs_json_parse(json_file)

        # 2. skip the empty image
        if len(objects) == 0:
            return

        # 3. obtain the info of building
        heights = np.array([obj['building_height'] for obj in objects])
        offsets = np.array([obj['offset'] for obj in objects])
        ignores = np.array([obj['ignore_flag'] for obj in objects])

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
        keep_flag, file_property = self.keep_file(angles, heights, offset_lengths, ignores)
        
        if keep_flag:
            mean_angle = file_property[0]
            self.save_count_results(mean_angle, file_name)
            return file_property
        else:
            return

    def keep_file(self, angles, heights, offset_lengths, ignores):
        if data_source == 'local':
            parameters = {
                        'angles': 0,
                        "mean_height": 0,
                        "mean_angle": 0, 
                        "mean_offset_length": 0,
                        'std_offset_length': 50,
                        'std_angle': 100,
                        'no_ignore_rate': 0.9}
        else:
            parameters = {
                        'angles': 15,
                        "mean_height": 4,
                        "mean_angle": 45, 
                        "mean_offset_length": 10,
                        'std_offset_length': 5,
                        'std_angle': 10,
                        'no_ignore_rate': 0.85}
        
        angles = np.abs(angles) * 180.0 / math.pi
        offset_lengths = np.abs(offset_lengths)

        mean_angle = np.mean(angles)
        mean_height = np.mean(heights)
        mean_offset_length = np.mean(offset_lengths)

        std_offset_length = np.std(offset_lengths)
        std_angle = np.std(angles)

        # parameters
        ignores = ignores.tolist()

        no_ignore_rate = ignores.count(0) / len(ignores)

        if no_ignore_rate < parameters['no_ignore_rate']:
            return False, None

        if len(angles) < parameters['angles']:
            return False, None

        if mean_height < parameters['mean_height']:
            return False, None

        if mean_angle < parameters['mean_angle'] and mean_offset_length < parameters['mean_offset_length'] and std_offset_length < parameters['std_offset_length']:
            return False, None

        if std_angle > parameters['std_angle']:
            return False, None

        file_property = [mean_angle, mean_height, mean_offset_length, std_offset_length, std_angle, no_ignore_rate]

        return True, file_property

    def core(self):
        json_file_list = glob.glob("{}/*.json".format(self.json_dir))

        mean_angles = []
        file_properties = []
        for json_file in tqdm.tqdm(json_file_list):
            file_name = bstool.get_basename(json_file)
            file_property = self.count_image(json_file)
            
            if file_property is not None:
                mean_angle = file_property[0]
                mean_angles.append(mean_angle)
                file_property.append(file_name)
                file_properties.append(file_property)
            else:
                continue
        
        return mean_angles, file_properties

    def save_count_results(self, angle, file_name):
        if math.isnan(angle):
            angle = 0 

        save_file = os.path.join(self.height_angle_save_dir, f"{int(angle / 5) * 5}.txt")
        with open(save_file, 'a+') as f:
            f.write(f'{self.city} {self.sub_fold} {file_name} {angle}\n')
        
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
    version = 'v2'

    data_source = args.source   # remote or local

    if data_source == 'local':
        cities = ['shanghai']
        sub_folds = {'shanghai': ['arg']}
        height_angle_save_dir = f'./data/buildchange/{version}/misc/height_angle'
        plt_save_dir = f'./data/buildchange/{version}/misc/plot'
        val_image_info_dir = None
        training_image_info_dir = f'./data/buildchange/{version}/misc/image_info'
    else:
        cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
        sub_folds = {'beijing':  ['arg', 'google', 'ms'],
                    'chengdu':  ['arg', 'google', 'ms'],
                    'haerbin':  ['arg', 'google', 'ms'],
                    'jinan':    ['arg', 'google', 'ms'],
                    'shanghai': ['arg', 'google', 'ms']}

        height_angle_save_dir = f'./data/buildchange/{version}/misc/height_angle'
        plt_save_dir = f'./data/buildchange/{version}/misc/plot'
        val_image_info_dir = '/mnt/lustrenew/liweijia/data/roof-footprint/paper/val_shanghai/'
        training_image_info_dir = f'./data/buildchange/{version}/misc/image_info'

    bstool.mkdir_or_exist(plt_save_dir)
    bstool.mkdir_or_exist(training_image_info_dir)
    if not os.path.exists(height_angle_save_dir):
        os.makedirs(height_angle_save_dir)
    else:
        shutil.rmtree(height_angle_save_dir)
        os.makedirs(height_angle_save_dir)
    
    full_mean_angles = []
    training_set = defaultdict(dict)
    for city in cities:
        full_mean_angles_city = []
        for sub_fold in sub_folds[city]:
            print("Begin processing {} {} set.".format(city, sub_fold))
            count_image = CountImage(core_dataset_name=core_dataset_name,
                                    version=version,
                                    city=city,
                                    sub_fold=sub_fold,
                                    height_angle_save_dir=height_angle_save_dir,
                                    val_image_info_dir=val_image_info_dir)
            mean_angles, file_properties = count_image.core()

            full_mean_angles += mean_angles
            full_mean_angles_city += mean_angles
            print("Finish processing {} {} set.".format(city, sub_fold))
            
            training_set[city][sub_fold] = file_properties

        full_mean_angles_city = np.array(full_mean_angles_city)
        plt.hist(full_mean_angles_city, bins=np.arange(0, 100, (int(100) - int(0)) // 20), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
        plt.savefig(os.path.join(plt_save_dir, f'test_{city}.png'))
        plt.clf()

    full_mean_angles = np.array(full_mean_angles)

    plt.hist(full_mean_angles, bins=np.arange(0, 100, (int(100) - int(0)) // 20), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
    plt.savefig(os.path.join(plt_save_dir, 'test_full_dataset.png'))
    plt.xlabel('angle')
    plt.ylabel('num')

    counter = defaultdict(dict)
    for city in cities:
        for sub_fold in sub_folds[city]:
            num = 0
            file_properties = training_set[city][sub_fold]
            with open(os.path.join(training_image_info_dir, f'{city}_{sub_fold}_select_image.txt'), 'w') as f:
                for file_property in file_properties:
                    file_property = [file_property[-1]] + ["{:.2f}".format(float(value)) for value in file_property[:-1]] + ['\n']
                    f.write(" ".join(file_property))
                    num += 1

            counter[city][sub_fold] = num

    print(f"The number of publiced image is: {counter}")
