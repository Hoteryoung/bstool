# -*- encoding: utf-8 -*-
'''
@File    :   vis_pkl_file_demo.py
@Time    :   2020/12/30 22:08:27
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   可视化 pkl 文件
'''


import os
import numpy as np
import mmcv
import bstool
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils


if __name__ == '__main__':
    # pkl_file = '/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/buildchange/bc_v001_mask_rcnn_r50_1x_v1_5city_trainval_roof_mask_roof_bbox/bc_v001_mask_rcnn_r50_1x_v1_5city_trainval_roof_mask_roof_bbox_coco_results.pkl'
    pkl_file = '/data/buildchange/bc_v001_mask_rcnn_r50_1x_v1_5city_trainval_roof_mask_roof_bbox_coco_results.pkl'
    ann_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_val_xian_fine.json'
    # ann_file = '/data/buildchange/buildchange_v1_val_xian_fine.json'
    image_dir = '/data/buildchange/v1/xian_fine/images'

    results = mmcv.load(pkl_file)

    coco = COCO(ann_file)
    img_ids = coco.get_img_ids()

    for idx, img_id in enumerate(img_ids):
        info = coco.load_imgs([img_id])[0]
        img_name = info['file_name']

        det, seg = results[idx]

        # for label in range(len(det)):
        # bboxes = det[label]
        bboxes = np.vstack(det)
        # print(len(det), len(bboxes), label)
        segms = mmcv.concat_list(seg)
        
        single_image_bbox = []
        single_image_mask = []
        single_image_score = []

        for i in range(bboxes.shape[0]):
            score = bboxes[i][4]
            if score < 0.5:
                continue

            if isinstance(segms[i]['counts'], bytes):
                segms[i]['counts'] = segms[i]['counts'].decode()
            
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            gray = np.array(mask*255, dtype=np.uint8)

            contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            
            if contours != []:
                cnt = max(contours, key = cv2.contourArea)
                if cv2.contourArea(cnt) < 5:
                    continue
                mask = np.array(cnt).reshape(1, -1).tolist()[0]
                if len(mask) < 8:
                    continue
            else:
                continue
            
            bbox = bboxes[i][0:4]
            score = bboxes[i][-1]

            single_image_bbox.append(bbox.tolist())
            single_image_mask.append(mask)
            single_image_score.append(score.tolist())

        img = cv2.imread(os.path.join(image_dir, img_name))

        bstool.show_bboxs_on_image(img, single_image_bbox)