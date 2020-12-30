# -*- encoding: utf-8 -*-
'''
@File    :   statistic_public_training.py
@Time    :   2020/12/30 22:21:14
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   对公开数据集进行统计分析
'''


import os
import csv
import numpy as np
from collections import defaultdict
import pandas
import tqdm
import shutil

import bstool


def parse_csv(csv_file):
    csv_df = pandas.read_csv(csv_file)
    file_names = list(csv_df.file_name.unique())

    return file_names

if __name__ == '__main__':
    csv_file = './data/buildchange/public/misc/nooverlap/training_dataset_info_20201028.csv'
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_folds = ['arg', 'google', 'ms']

    file_names = parse_csv(csv_file)

    statistic = defaultdict(int)
    for file_name in file_names:
        for city in cities:
            for sub_fold in sub_folds:
                if f'{city}_{sub_fold}' in file_name:
                    statistic[f'{city}_{sub_fold}'] += 1

    print(statistic)