# -*- encoding: utf-8 -*-
'''
@File    :   eval_height_demo.py
@Time    :   2020/12/30 21:53:42
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   对高度进行评估的 demo (可能无效)
'''


import bstool


if __name__ == '__main__':
    pkl_file = '/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/buildchange/bc_v006_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox/bc_v006_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_dalian_coco_results.pkl'
    anno_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_val_dalian_fine.json'
    pred_csv_prefix = '/data/buildchange/bc_v006_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_dalian_coco_results'

    # bstool.pkl2csv_roof_footprint(pkl_file, anno_file, pred_csv_prefix, score_threshold=0.05)

    mask_type = 'footprint'

    print(f"===== Start merge {mask_type} csv file =====")
    splitted_csv_file = pred_csv_prefix + f'_{mask_type}.csv'
    merged_csv_file = pred_csv_prefix + f"_{mask_type}_merged.csv"

    bstool.merge_csv_results_with_height(splitted_csv_file, merged_csv_file, iou_threshold=0.1, score_threshold=0.4, min_area=500)

