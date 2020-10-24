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

import bstool


class CountImage():
    def __init__(self,
                 core_dataset_name='buildchange',
                 src_version='v0',
                 dst_version='v2',
                 city='shanghai',
                 sub_fold=None,
                 resolution=0.6,
                 save_dir=None,
                 training_dir=None):
        self.city = city
        self.sub_fold = sub_fold
        self.resolution = resolution

        self.image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/images'
        if city == 'chengdu':
            self.json_dir = f'/mnt/lustre/menglingxuan/buildingwolf/20200329/{city}_L18/{sub_fold}/anno_20200924/OffsetField/TXT'
        else:
            if data_source == 'local':
                self.json_dir = '/data/buildchange/v0/shanghai/arg/json_20200924'
            else:
                self.json_dir = f'/mnt/lustre/menglingxuan/buildingwolf/20200329/{city}_18/{sub_fold}/anno_20200924/OffsetField/TXT'
            

        self.image_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}/images'
        bstool.mkdir_or_exist(self.image_save_dir)
        self.label_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}/labels'
        bstool.mkdir_or_exist(self.label_save_dir)

        self.save_dir = save_dir

        self.training_list = self.get_training_list(training_dir)

    def get_training_list(self, training_dir):
        if training_dir is None:
            return []

        training_list = []
        for sub_fold in os.listdir(training_dir):
            sub_dir = os.path.join(training_dir, sub_fold)
            for file_name in os.listdir(sub_dir):
                training_list.append(bstool.get_basename(file_name))

        training_list = list(set(training_list))

        return training_list

    def count_image(self, json_file):
        file_name = bstool.get_basename(json_file)
        
        if file_name in self.training_list:
            print(f"This image is in training list: {file_name}")
            return None

        image_file = os.path.join(self.image_dir, file_name + '.jpg')
        
        objects = bstool.lingxuan_json_parser(json_file)

        if len(objects) == 0:
            return None

        origin_properties = [obj['property'] for obj in objects]
        angles = []
        object_num = len(origin_properties)
        heights = []
        for single_property in origin_properties:
            
            if single_property['ignore'] == 1.0:
                continue
            
            if 'Floor' in single_property.keys():
                if single_property['Floor'] is None:
                    building_height = 0.0
                else:
                    building_height = 3 * single_property['Floor']
            elif 'half_H' in single_property.keys():
                if single_property['half_H'] is None:
                    building_height = 0.0
                else:
                    building_height = single_property['half_H']
            else:
                raise(RuntimeError("No Floor key in property, keys = {}".format(single_property.keys())))

            offset_x, offset_y = single_property['xoffset'], single_property['yoffset']

            angle = math.atan2(math.sqrt(offset_x ** 2 + offset_y ** 2) * self.resolution, building_height)
        
            angles.append(angle)
            heights.append(building_height)

        mean_angle = np.abs(np.mean(np.array(angles)) * 180.0 / math.pi)
        mean_height = np.mean(heights)

        if mean_height < 4:
            return None
        else:
            self.save_count_results(mean_angle, file_name)
            return mean_angle

    def core(self):
        json_file_list = glob.glob("{}/*.json".format(self.json_dir))

        mean_angles = []
        for json_file in tqdm.tqdm(json_file_list):
            mean_angle = self.count_image(json_file)
            if mean_angle is not None:
                mean_angles.append(mean_angle)
            else:
                continue
        
        return mean_angles

    def save_count_results(self, angle, file_name):
        save_file = os.path.join(self.save_dir, f"{int(angle / 5) * 5}.txt")
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
    src_version = 'v0'
    dst_version = 'v2'

    data_source = args.source   # remote or local

    if data_source == 'local':
        cities = ['shanghai']
        sub_folds = {'shanghai': ['arg']}
        save_dir = '/home/jwwangchn/Downloads/Count'
        plt_save_dir = '/home/jwwangchn/Downloads'
        training_dir = None
    else:
        cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
        sub_folds = {'beijing':  ['arg', 'google', 'ms'],
                    'chengdu':  ['arg', 'google', 'ms'],
                    'haerbin':  ['arg', 'google', 'ms'],
                    'jinan':    ['arg', 'google', 'ms'],
                    'shanghai': ['arg', 'google', 'ms']}

        save_dir = '/mnt/lustre/wangjinwang/Downloads/Count'
        plt_save_dir = '/mnt/lustre/wangjinwang/Downloads'
        training_dir = '/mnt/lustrenew/liweijia/data/roof-footprint/paper/val_shanghai/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    
    full_mean_angles = []
    for city in cities:
        full_mean_angles_city = []
        for sub_fold in sub_folds[city]:
            print("Begin processing {} {} set.".format(city, sub_fold))
            count_image = CountImage(core_dataset_name=core_dataset_name,
                                    src_version=src_version,
                                    dst_version=dst_version,
                                    city=city,
                                    sub_fold=sub_fold,
                                    save_dir=save_dir,
                                    training_dir=training_dir)
            mean_angles = count_image.core()

            full_mean_angles += mean_angles
            full_mean_angles_city += mean_angles
            print("Finish processing {} {} set.".format(city, sub_fold))

        full_mean_angles_city = np.array(full_mean_angles_city)
        plt.hist(full_mean_angles, bins=np.arange(0, 100, (int(100) - int(0)) // 20), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
        plt.savefig(os.path.join(plt_save_dir, f'test_{city}.png'))
        plt.clf()

    full_mean_angles = np.array(full_mean_angles)

    plt.hist(full_mean_angles, bins=np.arange(0, 100, (int(100) - int(0)) // 20), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
    plt.savefig(os.path.join(plt_save_dir, 'test_full_dataset.png'))
