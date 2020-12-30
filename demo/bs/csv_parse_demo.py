# -*- encoding: utf-8 -*-
'''
@File    :   csv_parse_demo.py
@Time    :   2020/12/30 21:52:09
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   解析 CSV 文件的 demo
'''


import bstool


if __name__ == '__main__':
    csv_file = '/data/buildchange/v0/xian_fine/xian_val_footprint_gt_minarea100_26.csv'

    csv_parser =  bstool.CSVParse(csv_file, 100)

    polygons = []
    for image_name in csv_parser.image_name_list:
        objects = csv_parser(image_name)

        polygons += [obj['polygon'] for obj in objects]

    print(len(polygons))