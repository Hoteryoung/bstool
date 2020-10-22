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

import bstool


class CountImage():
    def __init__(self,
                 core_dataset_name='buildchange',
                 src_version='v0',
                 dst_version='v2',
                 city='shanghai',
                 sub_fold=None,
                 resolution=0.6):
        self.city = city
        self.sub_fold = sub_fold
        self.resolution = resolution

        self.image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/images'
        if city == 'chengdu':
            self.json_dir = f'/mnt/lustre/menglingxuan/buildingwolf/20200329/{city}_L18/{sub_fold}/anno_20200924/OffsetField/TXT'
        else:
            self.json_dir = f'/mnt/lustre/menglingxuan/buildingwolf/20200329/{city}_18/{sub_fold}/anno_20200924/OffsetField/TXT'
            # self.json_dir = '/data/buildchange/v0/shanghai/arg/json_20200924'

        self.image_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}/images'
        bstool.mkdir_or_exist(self.image_save_dir)
        self.label_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}/labels'
        bstool.mkdir_or_exist(self.label_save_dir)

    def count_image(self, json_file):
        file_name = bstool.get_basename(json_file)

        image_file = os.path.join(self.image_dir, file_name + '.jpg')
        
        objects = bstool.lingxuan_json_parser(json_file)

        if len(objects) == 0:
            return

        origin_properties = [obj['property'] for obj in objects]
        angles = []
        for single_property in origin_properties:
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

        mean_angle = np.abs(np.mean(np.array(angles)) * 180.0 / math.pi)

        return mean_angle

    def core(self):
        json_file_list = glob.glob("{}/*.json".format(self.json_dir))

        mean_angles = []
        for json_file in tqdm.tqdm(json_file_list):
            mean_angle = self.count_image(json_file)
            mean_angles.append(mean_angle)
        
        return mean_angles


if __name__ == '__main__':
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    dst_version = 'v2'

    cities = ['shanghai']
    sub_folds = {'shanghai': ['arg']}

    # cities = ['beijing', 'jinan', 'haerbin', 'chengdu']                 # debug
    # cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    # sub_folds = {'beijing':  ['arg', 'google', 'ms', 'tdt'],
    #              'chengdu':  ['arg', 'google', 'ms', 'tdt'],
    #              'haerbin':  ['arg', 'google', 'ms'],
    #              'jinan':    ['arg', 'google', 'ms', 'tdt'],
    #              'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}

    full_mean_angles = []
    for city in cities:
        for sub_fold in sub_folds[city]:
            print("Begin processing {} {} set.".format(city, sub_fold))
            count_image = CountImage(core_dataset_name=core_dataset_name,
                                    src_version=src_version,
                                    dst_version=dst_version,
                                    city=city,
                                    sub_fold=sub_fold)
            mean_angles = count_image.core()

            full_mean_angles += mean_angles
            print("Finish processing {} {} set.".format(city, sub_fold))

    full_mean_angles = np.array(full_mean_angles)

    full_mean_angles = (full_mean_angles - np.min(full_mean_angles)) / (np.max(full_mean_angles) - np.min(full_mean_angles)) * 90

    plt.hist(full_mean_angles, bins=np.arange(0, 100, (int(100) - int(0)) // 10), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)

    plt.savefig('/mnt/lustre/wangjinwang/Downloads/test.png')
