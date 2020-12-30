# -*- encoding: utf-8 -*-
'''
@File    :   vis_wireframe_demo.py
@Time    :   2020/12/30 22:11:04
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   可视化 wireframe 信息
'''


import os
import numpy as np
import mmcv
import cv2
import matplotlib.pyplot as plt

import bstool


if __name__ == '__main__':
    wireframe_file = '/data/buildchange/v1/shanghai/wireframe/train.json'
    # wireframe_file = '/data/wireframe/data/wireframe/train.json'
    image_dir = '/data/buildchange/v1/shanghai/images'

    wireframes = mmcv.load(wireframe_file)

    for wireframe in wireframes:
        fig = plt.figure(figsize=(8, 8))

        filename = wireframe['filename']

        junctions = wireframe['junctions']
        positives = wireframe['edges_positive']
        negatives = wireframe['edges_negative']

        image_file = os.path.join(image_dir, filename)
        # img = cv2.imread(image_file)

        for positive in positives:
            start, end = positive

            plt.scatter(junctions[start][0], junctions[start][1])
            plt.scatter(junctions[end][0], junctions[end][1])
            plt.plot([junctions[start][0], junctions[end][0]], [junctions[start][1], junctions[end][1]])
        
        plt.axis([0, 1024, 0, 1024])
        plt.xlim(0, 1024)
        plt.ylim(0, 1024)
        plt.gca().invert_yaxis()
        plt.show()