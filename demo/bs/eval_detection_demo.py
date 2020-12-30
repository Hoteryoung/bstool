# -*- encoding: utf-8 -*-
'''
@File    :   eval_detection_demo.py
@Time    :   2020/12/30 21:53:00
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   使用 COCO API 进行检测相关的性能评估 demo
'''


import bstool


if __name__ == '__main__':
    pkl_file = '/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/buildchange/bc_v000_mask_rcnn_r50_1x_debug/bc_v000_mask_rcnn_r50_1x_debug_coco_results_merged.pkl'
    ann_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_train_samples_origin.json'
    json_prefix = 'bc_v000_mask_rcnn_r50_1x_debug'
    
    det_eval = bstool.DetEval(pkl_file, ann_file, json_prefix)
    det_eval.evaluation()