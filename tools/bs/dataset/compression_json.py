# -*- encoding: utf-8 -*-
'''
@File    :   compression_json.py
@Time    :   2020/12/30 22:13:55
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   压缩 json 文件，剔除部分重复的信息 (json 文件过大导致训练模型的时候内存溢出)
'''

import bstool
import mmcv
import json

if __name__ == '__main__':
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    for city in cities:
        src_json_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_train_{city}.json'
        dst_json_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_train_{city}_only_footprint.json'

        src_data = mmcv.load(src_json_file)
        src_annotations = src_data['annotations']

        dst_annotations = []
        for src_annotation in src_annotations:
            src_annotation['only_footprint'] = 1

            dst_annotations.append(src_annotation)

        src_data['annotations'] = dst_annotations

        with open(dst_json_file, "w") as jsonfile:
            json.dump(src_data, jsonfile, sort_keys=True, indent=4)