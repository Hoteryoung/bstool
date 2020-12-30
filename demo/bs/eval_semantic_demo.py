# -*- encoding: utf-8 -*-
'''
@File    :   eval_semantic_demo.py
@Time    :   2020/12/30 21:54:33
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   计算 F1 Score 的 demo
'''


import bstool


if __name__ == '__main__':
    pkl_file = '/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/buildchange/bc_v000_mask_rcnn_r50_1x_debug/bc_v000_mask_rcnn_r50_1x_debug_coco_results_merged.pkl'
    csv_gt_file = './data/buildchange/v0/samples/samples_2048_gt.csv'
    csv_pred_prefix = './pred'

    semantic_eval = bstool.SemanticEval(results=pkl_file,
                                        csv_gt_file=csv_gt_file,
                                        csv_pred_prefix=csv_pred_prefix,
                                        score_threshold=0.9)
    semantic_eval.evaluation()