import numpy as np
import cv2
import os
import argparse

import bstool


if __name__ == '__main__':
    image_root_dir = './data/buildchange/public/20201027/{}/images'
    label_root_dir = './data/buildchange/public/20201027/{}/labels'
    vis_root_dir = './data/buildchange/public/20201027/vis/{}'

    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_folds = {'beijing':  ['arg', 'google', 'ms'],
                'chengdu':  ['arg', 'google', 'ms'],
                'haerbin':  ['arg', 'google', 'ms'],
                'jinan':    ['arg', 'google', 'ms'],
                'shanghai': ['arg', 'google', 'ms']}

    for city in cities:
        for sub_fold in sub_folds[city]:
            image_dir = image_root_dir.format(city)
            label_dir = label_root_dir.format(city)
            vis_dir = vis_root_dir.format(city)

            bstool.mkdir_or_exist(vis_dir)

            for label_fn in os.listdir(label_dir):
                basename = bstool.get_basename(label_fn)
                
                label_file = os.path.join(label_dir, basename + '.json')
                image_file = os.path.join(image_dir, basename + '.png')

                objects = bstool.bs_json_parse(label_file)
                ignores = np.array([obj['ignore_flag'] for obj in objects])
                footprint_masks = np.array([obj['footprint_mask'] for obj in objects])
                roof_masks = np.array([obj['roof_mask'] for obj in objects])

                img = cv2.imread(image_file)

                img = bstool.draw_masks_boundary(img, footprint_masks, (0, 0, 255), thickness=1)
                img = bstool.draw_masks_boundary(img, roof_masks, (255, 0, 0), thickness=1)
                
                vis_file = os.path.join(vis_dir, basename + '.png')
                cv2.imwrite(vis_file, img)