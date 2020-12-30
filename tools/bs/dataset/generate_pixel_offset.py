# -*- encoding: utf-8 -*-
'''
@File    :   generate_pixel_offset.py
@Time    :   2020/12/30 22:16:06
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   生成 pixel-level 的 offset 标注数据
'''


import os
import numpy as np
import cv2
from skimage.draw import polygon
import tqdm

import bstool

from multiprocessing import Pool
from functools import partial


class Generator():
    def __init__(self,
                city,
                multi_processing=False,
                num_processor=8):
        self.label_dir = f'./data/buildchange/v1/{city}/labels'
        self.save_dir = f'./data/buildchange/v1/{city}/pixel_offset'
        bstool.mkdir_or_exist(self.save_dir)
        
        self.city = city
        self.multi_processing = multi_processing
        self.pool = Pool(num_processor)

    def generation(self, json_file):
        pixel_offset = np.zeros((1024, 1024, 2))
        file_name = bstool.get_basename(json_file)

        save_file = os.path.join(self.save_dir, file_name + '.npy')
        json_file = os.path.join(self.label_dir, file_name + '.json')

        objects = bstool.bs_json_parse(json_file)

        if len(objects) == 0:
            np.save(save_file, pixel_offset.astype(np.int16))
            return

        roof_masks = [obj['roof_mask'] for obj in objects]
        offsets = [obj['offset'] for obj in objects]

        for offset, roof_mask in zip(offsets, roof_masks):
            X, Y = polygon(roof_mask[0::2], roof_mask[1::2])
            
            X = np.clip(X, 0, 1023)
            Y = np.clip(Y, 0, 1023)

            pixel_offset[Y, X, 0] = int(offset[0])
            pixel_offset[Y, X, 1] = int(offset[1])

        np.save(save_file, pixel_offset.astype(np.int16))

    def core(self):
        label_file_list = os.listdir(self.label_dir)
        num = len(label_file_list)
        if self.multi_processing:
            worker = partial(self.generation)
            ret = list(tqdm.tqdm(self.pool.imap(worker, label_file_list), total=num))
        else:
            label_file_list = os.listdir(self.label_dir)
            for label_file in label_file_list:
                self.generation(label_file)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state) 


if __name__ == '__main__':
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']

    for city in cities:
        print("Begin processing {} set.".format(city))
        generator = Generator(city=city,
                              multi_processing=True,
                              num_processor=8)
        generator.core()
        print("Finish processing {} set.".format(city))