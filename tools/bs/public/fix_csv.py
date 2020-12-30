# -*- encoding: utf-8 -*-
'''
@File    :   fix_csv.py
@Time    :   2020/12/30 22:29:59
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   由于唯嘉在对图像进行分割的时候，子图像的坐标定义与我们这边不一样，导致错误，使用此程序进行修复她生成的 CSV 文件
'''


import bstool
import pandas
import shapely
import os
import cv2
import csv


def parse_csv(csv_file):
    full_data = []
    csv_df = pandas.read_csv(csv_file)

    for index in csv_df.index:
        data = csv_df.loc[index].values[0:].tolist()
        data[0] = data[0].split('__')[0] + '__' + data[0].split('__')[1].split('_')[1] + '_' + data[0].split('__')[1].split('_')[0]
        full_data.append(data)

    return full_data

def bs_csv_dump(full_data, csv_file):
    with open(csv_file, 'w') as summary:
        csv_writer = csv.writer(summary, delimiter=',')
        csv_writer.writerow(['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])
        for data in full_data:
            # if type(data[2]) == str:
            #     polygon = shapely.wkt.loads(data[2])
            # else:
            #     polygon = data[2]
            # if not bstool.single_valid_polygon(polygon):
            #     data[2] = polygon.buffer(0).wkt
            #     # continue
            csv_writer.writerow(data)

if __name__ == '__main__':
    for mask in ['roof', 'footprint']:
        
        src_csv_df = f'/data/buildchange/public/20201028/xian_val_{mask}_crop1024_gt_minarea500.csv'

        dst_csv_df = f'/data/buildchange/public/20201028/xian_val_{mask}_crop1024_gt_minarea500_fix.csv'

        full_data = parse_csv(src_csv_df)
        bs_csv_dump(full_data, dst_csv_df)




    