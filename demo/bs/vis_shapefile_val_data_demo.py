# -*- encoding: utf-8 -*-
'''
@File    :   vis_shapefile_val_data_demo.py
@Time    :   2020/12/30 22:09:06
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   可视化 shapefile 文件
'''


import bstool


if __name__ == '__main__':
    # shp_file = '/data/buildchange/v0/shanghai/shp_4326/L18_106968_219320.shp'
    shp_file = '/data/buildchange/v0/dalian_fine/merged_shp/dg_dalian__0_0.shp'
    geo_file = '/data/buildchange/v0/dalian_fine/images/dg_dalian__0_0.jpg'
    rgb_file = '/data/buildchange/v0/dalian_fine/images/dg_dalian__0_0.jpg'

    objects = bstool.shp_parse(shp_file=shp_file,
                                geo_file=geo_file,
                                src_coord='pixel',
                                dst_coord='pixel',
                                keep_polarity=False)

    polygons = [obj['polygon'] for obj in objects]
    masks = [obj['mask'] for obj in objects]
    print(masks)
    bboxes = [obj['bbox'] for obj in objects]

    bstool.show_polygon(polygons)
    bstool.show_masks_on_image(rgb_file, masks)
    bstool.show_bboxs_on_image(rgb_file, bboxes)