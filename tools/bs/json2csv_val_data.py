# -*- encoding: utf-8 -*-
'''
@File    :   json2csv_val_data.py
@Time    :   2020/12/30 22:39:04
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   将 json 转换为 CSV 文件
'''

import os
import bstool
import numpy as np
import rasterio as rio
import cv2
import pandas
import glob
from shapely import affinity
import tqdm


if __name__ == '__main__':
    roof_csv_file = './data/buildchange/v1/xian_fine/xian_fine_2048_roof_gt_splitted.csv'
    footprint_csv_file = './data/buildchange/v1/xian_fine/xian_fine_2048_footprint_gt_splitted.csv'
    
    first_in = True

    json_dir = './data/buildchange/v1/xian_fine/labels'

    json_file_list = glob.glob("{}/*.json".format(json_dir))
    for json_file in tqdm.tqdm(json_file_list):
        base_name = bstool.get_basename(json_file)

        objects = bstool.bs_json_parse(json_file)

                
        roof_gt_polygons, footprint_gt_polygons = [], []
        for obj in objects:
            roof_gt_polygon = bstool.mask2polygon(obj['roof_mask'])
            footprint_gt_polygon = bstool.mask2polygon(obj['footprint_mask'])
            
            foot_valid_flag = bstool.single_valid_polygon(roof_gt_polygon)
            footprint_valid_flag = bstool.single_valid_polygon(footprint_gt_polygon)

            if not (foot_valid_flag and footprint_valid_flag):
                continue

            roof_gt_polygons.append(roof_gt_polygon)
            footprint_gt_polygons.append(footprint_gt_polygon)

        roof_csv_image = pandas.DataFrame({'ImageId': base_name,
                                        'BuildingId': range(len(roof_gt_polygons)),
                                        'PolygonWKT_Pix': roof_gt_polygons,
                                        'Confidence': 1})
        footprint_csv_image = pandas.DataFrame({'ImageId': base_name,
                                        'BuildingId': range(len(footprint_gt_polygons)),
                                        'PolygonWKT_Pix': footprint_gt_polygons,
                                        'Confidence': 1})
        if first_in:
            roof_csv_dataset = roof_csv_image
            footprint_csv_dataset = footprint_csv_image
            first_in = False
        else:
            roof_csv_dataset = roof_csv_dataset.append(roof_csv_image)
            footprint_csv_dataset = footprint_csv_dataset.append(footprint_csv_image)

    roof_csv_dataset.to_csv(roof_csv_file, index=False)
    footprint_csv_dataset.to_csv(footprint_csv_file, index=False)