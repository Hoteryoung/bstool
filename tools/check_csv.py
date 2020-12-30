# -*- encoding: utf-8 -*-
'''
@File    :   check_csv.py
@Time    :   2020/12/30 22:43:04
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   检查 CSV 文件的合法性
'''


import bstool
import pandas
import shapely
import os
import cv2


if __name__ == '__main__':
    csv_file = '/data/urban3d/v1/ATL/urban3d_atl_roof_offset_gt_simple_subcsv_merge_val.csv'

    csv_parser = bstool.CSVParse(csv_file)

    for image_fn in csv_parser.image_fns:
        csv_parser(image_fn)