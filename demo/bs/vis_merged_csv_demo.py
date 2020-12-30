# -*- encoding: utf-8 -*-
'''
@File    :   vis_merged_csv_demo.py
@Time    :   2020/12/30 22:07:47
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   可视化 CSV 文件合并 demo
'''


import bstool
import pandas
import shapely
import os
import cv2


if __name__ == '__main__':

    image_dir = '/data/buildchange/v1/xian_fine_origin/images'
    csv_df = pandas.read_csv('/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/bc_v015_mask_rcnn_r50_v2_roof_trainval/bc_v015_mask_rcnn_r50_v2_roof_trainval_footprint_merged.csv')

    for image_name in os.listdir(image_dir):
        image_file = os.path.join(image_dir, image_name)
        image_basename = bstool.get_basename(image_name)

        img = cv2.imread(image_file)
 
        roof_masks = []
        for idx, row in csv_df[csv_df.ImageId == image_basename].iterrows():
            roof_polygon = shapely.wkt.loads(row.PolygonWKT_Pix)

            roof_mask = bstool.polygon2mask(roof_polygon)
            roof_masks.append(roof_mask)

        if len(roof_masks) == 0:
            continue
        
        img = bstool.draw_mask_boundary(img, roof_masks)
        bstool.show_image(img)