import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas
import cv2
import glob
from multiprocessing import Pool
from functools import partial
import tqdm

import bstool


class SplitImage():
    def __init__(self,
                 core_dataset_name='buildchange',
                 src_version='v0',
                 dst_version='v1',
                 city='shanghai',
                 sub_fold=None,
                 subimage_size=1024,
                 gap=512,
                 multi_processing=False,
                 num_processor=8):
        self.city = city
        self.sub_fold = sub_fold
        self.subimage_size = subimage_size
        self.gap = gap

        self.offset_field_dir = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/offset_field/Npy'

        wrong_file = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/wrongShpFile.txt'
        self.skip_filenames = self.read_wrong_file(wrong_file)

        self.label_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}/offset_field'
        bstool.mkdir_or_exist(self.label_save_dir)

        self.multi_processing = multi_processing
        self.pool = Pool(num_processor)
    
    def read_wrong_file(self, wrong_file):
        skip_filenames = []
        with open(wrong_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                skip_filename = bstool.get_basename(line.strip('\n'))
                skip_filenames.append(skip_filename)
        
        return skip_filenames

    def split_image(self, image_file):
        file_name = bstool.get_basename(image_file)
        if file_name in self.skip_filenames:
            return

        img = np.load(image_file)

        subimages = bstool.split_image(img, 
                                        subsize=self.subimage_size, 
                                        gap=self.gap)
        subimage_coordinates = list(subimages.keys())
        
        for subimage_coordinate in subimage_coordinates:

            subimage = subimages[subimage_coordinate]

            subimage_file = os.path.join(self.label_save_dir, f'{self.city}_{self.sub_fold}__{file_name}__{subimage_coordinate[0]}_{subimage_coordinate[1]}.npy')

            np.save(subimage_file, subimage)

    def core(self):
        offset_field_list = glob.glob("{}/*.npy".format(self.offset_field_dir))
        num_image = len(offset_field_list)
        if self.multi_processing:
            worker = partial(self.split_image)
            ret = list(tqdm.tqdm(self.pool.imap(worker, offset_field_list), total=num_image))
            self.pool.close()
            self.pool.join()
        else:
            offset_field_list = glob.glob("{}/*.npy".format(self.offset_field_dir))
            for image_file in tqdm.tqdm(offset_field_list):
                self.split_image(image_file)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state) 


if __name__ == '__main__':
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    dst_version = 'v1'

    # cities = ['shanghai']
    # sub_folds = {'shanghai': ['arg']}

    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_folds = {'beijing':  ['arg', 'google', 'ms', 'tdt'],
                 'chengdu':  ['arg', 'google', 'ms', 'tdt'],
                 'haerbin':  ['arg', 'google', 'ms'],
                 'jinan':    ['arg', 'google', 'ms', 'tdt'],
                 'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}
    
    subimage_size = 1024
    gap = subimage_size // 2

    for city in cities:
        for sub_fold in sub_folds[city]:
            print("Begin processing {} {} set.".format(city, sub_fold))
            split_image = SplitImage(core_dataset_name=core_dataset_name,
                                    src_version=src_version,
                                    dst_version=dst_version,
                                    city=city,
                                    sub_fold=sub_fold,
                                    subimage_size=subimage_size,
                                    gap=gap,
                                    multi_processing=True,
                                    num_processor=2)
            split_image.core()
            print("Finish processing {} {} set.".format(city, sub_fold))
