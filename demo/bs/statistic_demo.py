# -*- encoding: utf-8 -*-
'''
@File    :   statistic_demo.py
@Time    :   2020/12/30 22:02:23
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   计算数据统计信息的 demo
'''

import bstool


if __name__ == '__main__':
    
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu', 'dalian_fine', 'xian_fine']
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    cities = ['xian_fine']
    csv_files = []
    title = []
    for city in cities:
        print("City: ", city)
        
        csv_file = f'./data/buildchange/v0/{city}/{city}_2048_footprint_gt.csv'
        csv_files.append(csv_file)
        title.append(city)

        statistic = bstool.Statistic(ann_file=None, csv_file=csv_files)
        
        # statistic.height_distribution(title)
        # statistic.height_curve(title)

        statistic.offset_distribution(title)
        statistic.offset_polar(title)
