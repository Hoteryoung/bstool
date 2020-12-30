# -*- encoding: utf-8 -*-
'''
@File    :   convert2wireframe_demo.py
@Time    :   2020/12/30 21:51:42
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   将 mask 转换为 wireframw 的 demo
'''


import os
import numpy as np
import mmcv
import json

import bstool


if __name__ == '__main__':
    image_dir = '/data/buildchange/v1/shanghai/images'
    label_dir = '/data/buildchange/v1/shanghai/labels'
    json_file = '/data/buildchange/v1/shanghai/wireframe/train.json'

    height, width = 1024, 1024

    masks = [[100, 100, 200, 200, 300, 300, 400, 400], [500, 500, 600, 600, 700, 700, 800, 800]]

    wireframe_items = bstool.mask2wireframe(masks)
    junctions = wireframe_items[0]
    edges_positive = wireframe_items[1]
    edges_negative = wireframe_items[2]

    print(junctions, edges_positive, edges_negative)