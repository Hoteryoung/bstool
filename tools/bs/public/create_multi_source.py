# -*- encoding: utf-8 -*-
'''
@File    :   create_multi_source.py
@Time    :   2020/12/30 22:26:15
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   根据数据源来对数据进行划分
'''

import os
import shutil
import bstool


if __name__ == '__main__':
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_folds = ['arg', 'google', 'ms']

    src_image_root = './data/buildchange/public/20201028/{}/images'
    src_label_root = './data/buildchange/public/20201028/{}/labels'

    for sub_fold in sub_folds:
        for city in cities:
            src_image_dir = src_image_root.format(city)
            src_label_dir = src_label_root.format(city)
            dst_image_dir = src_image_root.format(sub_fold)
            dst_label_dir = src_label_root.format(sub_fold)
            
            bstool.mkdir_or_exist(dst_image_dir)
            bstool.mkdir_or_exist(dst_label_dir)

            for fn in os.listdir(src_image_dir):
                basename = bstool.get_basename(fn)

                if sub_fold in basename:
                    src_image = os.path.join(src_image_dir, basename + '.png')
                    src_label = os.path.join(src_label_dir, basename + '.json')

                    dst_image = os.path.join(dst_image_dir, basename + '.png')
                    dst_label = os.path.join(dst_label_dir, basename + '.json')

                    shutil.copy(src_image, dst_image)
                    shutil.copy(src_label, dst_label)
            
