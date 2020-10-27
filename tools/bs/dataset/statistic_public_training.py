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
    csv_file = './data/buildchange/public/misc/nooverlap/training_dataset_info.csv'
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