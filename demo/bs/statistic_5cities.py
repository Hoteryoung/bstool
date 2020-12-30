# -*- encoding: utf-8 -*-
'''
@File    :   statistic_5cities.py
@Time    :   2020/12/30 22:01:56
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   生成 5 个城市的统计数据
'''

import bstool


if __name__ == '__main__':
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    csv_files = []
    title = []
    for city in cities:
        print("City: ", city)
        
        csv_file = f'./data/buildchange/v0/{city}/{city}_2048_footprint_gt.csv'
        csv_files.append(csv_file)
        title.append(city)

    statistic = bstool.Statistic(ann_file=None, csv_file=csv_files)
    statistic.offset_polar(title)
    statistic.height_distribution(title)
