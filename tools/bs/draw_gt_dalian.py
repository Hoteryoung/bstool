
# -*- encoding: utf-8 -*-
'''
@File    :   draw_gt_dalian.py
@Time    :   2020/12/19 23:07:16
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   可视化 shapefile 格式的 dalian gt 数据
'''


import os
import bstool
import numpy as np
import rasterio as rio
import cv2
import pandas
import glob
from shapely import affinity


if __name__ == '__main__':
    first_in = True
    min_area = 100

    shp_dir = f'./data/buildchange/v0/dalian_fine/merged_shp'
    rgb_img_dir = f'./data/buildchange/v0/dalian_fine/images'
    output_dir = '/data/buildchange/v0/dalian_fine/vis/gt'

    bstool.mkdir_or_exist(output_dir)

    shp_file_list = glob.glob("{}/*.shp".format(shp_dir))
    for shp_file in shp_file_list:
        base_name = bstool.get_basename(shp_file)

        rgb_img_file = os.path.join(rgb_img_dir, base_name + '.jpg')
        img = cv2.imread(rgb_img_file)

        objects = bstool.shp_parse(shp_file=shp_file,
                                    geo_file=rgb_img_file,
                                    src_coord='pixel',
                                    dst_coord='pixel',
                                    keep_polarity=False)

        roof_masks = []
        for obj in objects:
            roof_gt_polygon = obj['polygon']

            valid_flag = bstool.single_valid_polygon(roof_gt_polygon)
            if not valid_flag:
                continue

            roof_masks.append(bstool.polygon2mask(obj['polygon']))

        output_file = os.path.join(output_dir, base_name + '.jpg')
        img = bstool.draw_masks_boundary(img, roof_masks)

        cv2.imwrite(output_file, img)
