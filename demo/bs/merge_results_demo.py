import os
import numpy as np
import mmcv
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import cv2
from collections import defaultdict
from tqdm import tqdm

import bstool


if __name__ == '__main__':
    large_image_dir = '/data/buildchange/v1/xian_fine_origin/images'
    anno_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_val_xian_fine.json'
    pkl_file = '/data/buildchange/bc_v001_mask_rcnn_r50_1x_v1_5city_trainval_roof_mask_roof_bbox_coco_results.pkl'
    
    results = mmcv.load(pkl_file)
    ret = bstool.merge_results(results, anno_file)

    for image_file_name in os.listdir(large_image_dir):
        image_file = os.path.join(large_image_dir, image_file_name)
        img = cv2.imread(image_file)

        nmsed_bboxes, nmsed_masks, nmsed_scores = ret[bstool.get_basename(image_file)]

        bstool.show_bboxs_on_image(img, nmsed_bboxes, show=True)
        bstool.show_masks_on_image(img, nmsed_masks, show=True)

        