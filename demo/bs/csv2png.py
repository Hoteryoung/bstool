# -*- encoding: utf-8 -*-
'''
@File    :   csv2png.py
@Time    :   2020/12/30 21:52:33
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   可视化 CSV 文件的 demo
'''


import os
import numpy as np
import cv2

import bstool


if __name__ == '__main__':
    csv_file = '/data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt.csv'
    output_dir = '/data/buildchange/v0/xian_fine/png_gt'

    csv_parser = bstool.CSVParse(csv_file)

    image_name_list = csv_parser.image_name_list
    
    for image_name in image_name_list:
        objects = csv_parser(image_name)

        polygons = [obj['polygon'] for obj in objects]


        foreground = bstool.generate_image(2048, 2048, (0, 0, 0))
        for idx, polygon in enumerate(polygons):
            mask = bstool.polygon2mask(polygon)
            mask = np.array(mask).reshape(1, -1, 2)
            cv2.fillPoly(foreground, mask, (255, 255, 255))

        output_file = os.path.join(output_dir, image_name + '.png')
        print(output_file)
        cv2.imwrite(output_file, foreground)