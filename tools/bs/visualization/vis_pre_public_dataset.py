# -*- encoding: utf-8 -*-
'''
@File    :   vis_pre_public_dataset.py
@Time    :   2020/12/30 22:35:37
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   对公开数据集进行可视化
'''


import numpy as np
import cv2
import os
import argparse

import bstool


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
    data_source = args.source   # remote or local

    if data_source == 'local':
        cities = ['shanghai']
        sub_folds = {'shanghai': ['arg']}
        images_dir = './data/buildchange/v2/{}/images'
        labels_dir = './data/buildchange/v2/{}/labels'
        vis_dir_ori = './data/buildchange/v2/misc/nooverlap/vis/{}'
        image_info_dir = './data/buildchange/v2/misc/nooverlap/image_info'
    else:
        cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
        sub_folds = {'beijing':  ['arg', 'google', 'ms'],
                    'chengdu':  ['arg', 'google', 'ms'],
                    'haerbin':  ['arg', 'google', 'ms'],
                    'jinan':    ['arg', 'google', 'ms'],
                    'shanghai': ['arg', 'google', 'ms']}

        images_dir = './data/buildchange/v2/{}/images'
        labels_dir = './data/buildchange/v2/{}/labels'
        vis_dir_ori = './data/buildchange/v2/misc/vis/{}'
        image_info_dir = './data/buildchange/v2/misc/image_info'

    

    for city in cities:
        for sub_fold in sub_folds[city]:
            image_dir = images_dir.format(city)
            label_dir = labels_dir.format(city)
            vis_dir = vis_dir_ori.format(city)

            bstool.mkdir_or_exist(vis_dir)

            imageset_file = os.path.join(image_info_dir, f'{city}_{sub_fold}_select_image.txt')

            with open(imageset_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                image_name = line.strip().split(' ')[0]
            
                image_file = os.path.join(image_dir, image_name + '.png')
                label_file = os.path.join(label_dir, image_name + '.json')

                if (not os.path.exists(image_file)) or (not os.path.exists(label_file)):
                    print("This file is not exist: {}".format(image_name))
                    continue

                objects = bstool.bs_json_parse(label_file)
                ignores = np.array([obj['ignore_flag'] for obj in objects])
                footprint_masks = np.array([obj['footprint_mask'] for obj in objects])
                roof_masks = np.array([obj['roof_mask'] for obj in objects])

                img = cv2.imread(image_file)

                img = bstool.draw_masks_boundary(img, footprint_masks, (0, 0, 255), thickness=1)
                img = bstool.draw_masks_boundary(img, roof_masks, (255, 0, 0), thickness=1)
                
                vis_file = os.path.join(vis_dir, image_name + '.png')
                cv2.imwrite(vis_file, img)