# -*- encoding: utf-8 -*-
'''
@File    :   offset_distribution.py
@Time    :   2020/12/30 22:32:13
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   计算 offset 的统计分布
'''


import os
import cv2
import numpy as np
import bstool
from matplotlib import pyplot as plt


if __name__ == '__main__':
    cities = ['beijing', 'shanghai', 'chengdu', 'jinan', 'haerbin']

    label_root = '/data/buildchange/public/20201028/{}/labels'
    
    offsets = []
    for city in cities:
        label_dir = label_root.format(city)
        for image_name in os.listdir(label_dir):
            file_name = bstool.get_basename(image_name)
            json_file = os.path.join(label_dir, file_name + '.json')

            objects = bstool.urban3d_json_parse(json_file)

            if len(objects) == 0:
                continue

            offsets += [bstool.offset_coordinate_transform(obj['offset']) for obj in objects]
       
    offsets = np.array(offsets)

    angles = offsets[:, 1]

    plt.hist(angles, bins=np.arange(-np.pi, np.pi, np.pi/18), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)

    plt.show()