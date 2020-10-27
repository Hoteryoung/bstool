import numpy as np
import cv2
import os
import argparse
import random
import shutil
import pandas

import bstool


def parse_csv(csv_file):
    csv_df = pandas.read_csv(csv_file)
    file_names = list(csv_df.file_name.unique())
    scores = list(csv_df.score.unique())

    return file_names, scores

if __name__ == '__main__':
    vis_root_dir = './data/buildchange/public/20201027/vis/{}'
    csv_file = './data/buildchange/public/misc/nooverlap/training_dataset_info_20201027.csv'

    file_names, scores = parse_csv(csv_file)

    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_folds = {'beijing':  ['arg', 'google', 'ms'],
                'chengdu':  ['arg', 'google', 'ms'],
                'haerbin':  ['arg', 'google', 'ms'],
                'jinan':    ['arg', 'google', 'ms'],
                'shanghai': ['arg', 'google', 'ms']}

    vis_file_list = []
    for city in cities:
        for sub_fold in sub_folds[city]:
            vis_dir = vis_root_dir.format(city)
            for image_fn in os.listdir(vis_dir):
                basename = bstool.get_basename(image_fn)
                vis_file = os.path.join(vis_dir, basename + '.png')
                vis_file_list.append(vis_file)

    with_random = False
    if with_random:
        random.seed(0)
        vis_file_list.sort()
        random.shuffle(vis_file_list)
        random_vis_dir = './data/buildchange/public/20201027/random_vis'
    else:
        sorted_index = np.argsort(scores)
        vis_file_list = [vis_file_list[idx] for idx in sorted_index]
        random_vis_dir = './data/buildchange/public/20201027/high_score_image_300'

    bstool.mkdir_or_exist(random_vis_dir)
    
    max_score, min_score = max(scores), min(scores)

    for vis_file in vis_file_list[0:300]:
        basename = bstool.get_basename(vis_file)

        score = (scores[file_names.index(basename)] - min_score) / (max_score - min_score) * 100 + 100

        shutil.copy(vis_file, os.path.join(random_vis_dir, str(int(score)) + '_' + basename))

    
