# -*- encoding: utf-8 -*-
'''
@File    :   generate_empty_edge_map.py
@Time    :   2020/12/30 22:37:33
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   edge map 是唯嘉生成的，数量上与我这面有些不对应，为了能够正常训练网络，生成了一些空白的 edge map
'''


import os
import cv2
import bstool


if __name__ == '__main__':
    core_dataset_name = 'buildchange'

    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']

    src_version = 'v1'

    for city in cities:
        image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/images'
        edge_image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/edge_labels'

        edge_image_list = os.listdir(edge_image_dir)
        image_list = os.listdir(image_dir)

        for image_name in image_list:
            if image_name not in edge_image_list:
                empty_edge_map = bstool.generate_image(1024, 1024, 0)
                edge_file = os.path.join(edge_image_dir, image_name)
                print(f"generate empty edge image: {edge_file}")
                cv2.imwrite(edge_file, empty_edge_map)
            
