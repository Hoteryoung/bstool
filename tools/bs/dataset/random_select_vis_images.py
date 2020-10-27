import numpy as np
import cv2
import os
import argparse
import random
import shutil

import bstool


if __name__ == '__main__':
    vis_root_dir = './data/buildchange/public/20201027/vis/{}'
    random_vis_dir = './data/buildchange/public/20201027/random_vis'

    bstool.mkdir_or_exist(random_vis_dir)

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

    
    random.seed(0)
    vis_file_list.sort()
    random.shuffle(vis_file_list)

    for vis_file in vis_file_list[0:300]:
        basename = bstool.get_basename(vis_file)
        shutil.copy(vis_file, os.path.join(random_vis_dir, basename))

    
