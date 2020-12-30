# -*- encoding: utf-8 -*-
'''
@File    :   parse_csv_demo.py
@Time    :   2020/12/30 22:00:09
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   解析 CSV 文件的 demo
'''


import os
import cv2

import bstool


if __name__ == '__main__':
    gt_footprint_csv_file = '/data/buildchange/v0/dalian_fine/dalian_fine_2048_footprint_gt.csv'

    csv_parser = bstool.CSVParse(gt_footprint_csv_file)

    for image_name in csv_parser.image_name_list:
        objects = csv_parser(image_name)

        print(objects)