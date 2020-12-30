# -*- encoding: utf-8 -*-
'''
@File    :   parse_pkl_demo.py
@Time    :   2020/12/30 22:00:33
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   解析 pkl 文件的 demo
'''


import bstool
import pandas
import shapely
import os
import cv2


if __name__ == '__main__':
    image_set = 'dalian_fine'

    anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_val_{image_set}.json'
    pkl_file = f'/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/buildchange/bc_v006_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox/bc_v006_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_dalian_coco_results.pkl'
    
    origin_image_dir = f'/data/buildchange/v0/{image_set}/images'
    splitted_image_dir = f'/data/buildchange/v1/{image_set}/images'

    pkl_parser = bstool.BSPklParser(anno_file, pkl_file, score_threshold=0.4, with_offset=True, with_height=True)

    # for image_name in os.listdir(splitted_image_dir):
    for image_name in os.listdir(origin_image_dir):
        image_base_name = bstool.get_basename(image_name)
        image_file = os.path.join(origin_image_dir, image_name)

        buildings = pkl_parser(image_base_name, splitted=False)

        if len(buildings) == 0:
            continue

        roof_polygons = [building['footprint_polygon'] for building in buildings]

        bstool.show_polygons_on_image(image_file, roof_polygons)