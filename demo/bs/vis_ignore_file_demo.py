# -*- encoding: utf-8 -*-
'''
@File    :   vis_ignore_file_demo.py
@Time    :   2020/12/30 22:06:19
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   可视化 ignore 标志位
'''

import bstool


if __name__ == '__main__':
    shp_file = './data/buildchange/v0/shanghai/merged_shp/L18_106968_219320.shp'
    ignore_file = './data/buildchange/v0/shanghai/anno_v2/L18_106968_219320.png'
    geo_file = './data/buildchange/v0/shanghai/geo_info/L18_106968_219320.png'
    rgb_file = './data/buildchange/v0/shanghai/images/L18_106968_219320.jpg'

    objects = bstool.mask_parse(ignore_file, subclasses=255)

    polygons = [obj['polygon'] for obj in objects]
    masks = [obj['mask'] for obj in objects]

    bstool.show_polygon(polygons)
    bstool.show_masks_on_image(rgb_file, masks)