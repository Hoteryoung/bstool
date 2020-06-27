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
    large_image_dir = '/data/buildchange/v0/xian_fine'
    subimage_dir = '/home/jwwangchn/Documents/100-Work/170-Codes/aidet/data/buildchange/v2/xian_fine/images'
    model = 'bc_v015_mask_rcnn_r50_v2_roof_trainval'
    
    anno_file = '/home/jwwangchn/Documents/100-Work/170-Codes/aidet/data/buildchange/v2/coco/annotations/buildchange_v2_val_xian_fine.json'
    results = mmcv.load(f'/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/{model}/coco_results.pkl')

    coco = COCO(anno_file)
    catIds = coco.getCatIds(catNms=[''])
    imgIds = coco.getImgIds(catIds=catIds)

    # merged_results = defaultdict(list)
    merged_bboxes = defaultdict(dict)
    merged_masks = defaultdict(dict)
    merged_scores = defaultdict(dict)
    subfolds = {}

    for idx, imgId in tqdm(enumerate(imgIds)):
        img = coco.loadImgs(imgIds[idx])[0]
        img_name = img['file_name']

        base_name = bstool.get_basename(img_name)
        sub_fold = base_name.split("__")[0].split('_')[0]
        ori_image_fn = "_".join(base_name.split("__")[0].split('_')[1:])
        coord_x, coord_y = base_name.split("__")[1].split('_')    # top left corner
        coord_x, coord_y = int(coord_x), int(coord_y)

        single_image_results = {}
        det, seg, offset = results[idx]

        for label in range(len(det)):
            bboxes = det[label]
            if isinstance(seg, tuple):
                segms = seg[0][label]
            else:
                segms = seg[label]

            if isinstance(offset, tuple):
                offsets = offset[0]
            else:
                offsets = offset
            
            single_image_bbox = []
            single_image_mask = []
            single_image_score = []
            for i in range(bboxes.shape[0]):
                score = bboxes[i][4]
                if score < 0.3:
                    continue
                offset = offsets[i]

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

        # single_image_results[(coord_x, coord_y)] = single_image_bbox

        subfolds[ori_image_fn] = sub_fold
        
        # merged_results[ori_image_fn].append(single_image_results)
        merged_bboxes[ori_image_fn][(coord_x, coord_y)] = np.array(single_image_bbox)
        merged_masks[ori_image_fn][(coord_x, coord_y)] = np.array(single_image_mask)
        merged_scores[ori_image_fn][(coord_x, coord_y)] = np.array(single_image_score)

    for ori_image_fn, sub_fold in subfolds.items():
        ori_image_file = os.path.join(large_image_dir, sub_fold, 'images', ori_image_fn + '.jpg')

        img = cv2.imread(ori_image_file)

        ori_image_bboxes = merged_bboxes[ori_image_fn]
        ori_image_masks = merged_masks[ori_image_fn]
        ori_image_scores = merged_scores[ori_image_fn]

        nmsed_bboxes, nmsed_masks = bstool.merge_results_on_subimage((ori_image_bboxes, ori_image_masks, ori_image_scores), 
                                                                     iou_threshold=0.1)

        bstool.show_bboxs_on_image(img, nmsed_bboxes, show=True)
        bstool.show_masks_on_image(img, nmsed_masks, show=True)

        