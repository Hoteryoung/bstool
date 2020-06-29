import os
import numpy as np
import mmcv
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import cv2
from collections import defaultdict
from tqdm import tqdm

import bstool

try:
    import solaris
except:
    print("Please install solaris")


def merge_results_on_subimage(results_with_coordinate, iou_threshold=0.5, nms='bbox_nms'):
    """designed for bboxes and masks

    Args:
        results_with_coordinate ([type]): [description]
        iou_threshold (float, optional): [description]. Defaults to 0.5.

    Returns:
        [type]: [description]
    """
    if isinstance(results_with_coordinate, tuple):
        if len(results_with_coordinate) == 2:
            bboxes_with_coordinate, scores_with_coordinate = results_with_coordinate
            masks_with_coordinate = None
        elif len(results_with_coordinate) == 3:
            bboxes_with_coordinate, masks_with_coordinate, scores_with_coordinate = results_with_coordinate
        else:
            raise(RuntimeError("wrong len of results_with_coordinate: ", len(results_with_coordinate)))
    
    subimage_coordinates = list(bboxes_with_coordinate.keys())

    bboxes_merged = []
    masks_merged = []
    scores_merged = []
    for subimage_coordinate in subimage_coordinates:
        bboxes_single_image = bboxes_with_coordinate[subimage_coordinate]
        masks_single_image = masks_with_coordinate[subimage_coordinate]
        scores_single_image = scores_with_coordinate[subimage_coordinate]

        if len(bboxes_single_image) == 0:
            continue

        bboxes_single_image = bstool.chang_bbox_coordinate(bboxes_single_image, subimage_coordinate)
        masks_single_image = bstool.chang_mask_coordinate(masks_single_image, subimage_coordinate)

        bboxes_merged += bboxes_single_image.tolist()
        masks_merged += masks_single_image
        scores_merged += scores_single_image.tolist()

    if nms == 'bbox_nms':
        keep = bstool.bbox_nms(np.array(bboxes_merged), np.array(scores_merged), iou_threshold=iou_threshold)
    elif nms == 'mask_nms':
        keep = bstool.mask_nms(masks_merged, np.array(scores_merged), iou_threshold=iou_threshold)
    else:
        keep = range(len(bboxes_merged))

    return np.array(bboxes_merged)[keep].tolist(), np.array(masks_merged)[keep], np.array(scores_merged)[keep].tolist()

def merge_results(results, anno_file, iou_threshold=0.1, score_threshold=0.05, nms='bbox_nms', opencv_flag=True):
    coco = COCO(anno_file)
    img_ids = coco.get_img_ids()

    merged_bboxes = defaultdict(dict)
    merged_masks = defaultdict(dict)
    merged_scores = defaultdict(dict)
    subfolds = {}

    for idx, img_id in tqdm(enumerate(img_ids)):
        info = coco.load_imgs([img_id])[0]
        img_name = info['file_name']

        base_name = bstool.get_basename(img_name)
        sub_fold = base_name.split("__")[0].split('_')[1]
        ori_image_fn = base_name.split("__")[1]
        coord_x, coord_y = base_name.split("__")[2].split('_')    # top left corner
        coord_x, coord_y = int(coord_x), int(coord_y)

        det, seg = results[idx]

        bboxes = np.vstack(det)
        segms = mmcv.concat_list(seg)
        
        single_image_bbox = []
        single_image_mask = []
        single_image_score = []
        for i in range(bboxes.shape[0]):
            score = bboxes[i][4]
            if score < score_threshold:
                continue

            if isinstance(segms[i]['counts'], bytes):
                segms[i]['counts'] = segms[i]['counts'].decode()
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            gray = np.array(mask * 255, dtype=np.uint8)

            if opencv_flag:
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
            else:
                polygons = bstool.generate_polygon(gray)
                if len(polygons) == 0:
                    continue
                areas = [polygon.area for polygon in polygons]
                idx = areas.index(max(areas))
                mask = bstool.polygon2mask(polygons[idx])
                
            bbox = bboxes[i][0:4]
            score = bboxes[i][-1]

            single_image_bbox.append(bbox.tolist())
            single_image_mask.append(mask)
            single_image_score.append(score.tolist())

        subfolds[ori_image_fn] = sub_fold
        
        merged_bboxes[ori_image_fn][(coord_x, coord_y)] = np.array(single_image_bbox)
        merged_masks[ori_image_fn][(coord_x, coord_y)] = np.array(single_image_mask)
        merged_scores[ori_image_fn][(coord_x, coord_y)] = np.array(single_image_score)

    ret = {}
    for ori_image_fn, sub_fold in subfolds.items():
        ori_image_bboxes = merged_bboxes[ori_image_fn]
        ori_image_masks = merged_masks[ori_image_fn]
        ori_image_scores = merged_scores[ori_image_fn]

        nmsed_bboxes, nmsed_masks, nmsed_scores = bstool.merge_results_on_subimage((ori_image_bboxes, ori_image_masks, ori_image_scores), 
                                                                     iou_threshold=iou_threshold, nms='bbox_nms')
        ret[ori_image_fn] = (nmsed_bboxes, nmsed_masks, nmsed_scores)

    return ret

def solaris_semantic_evaluation(csv_pred_file, csv_gt_file):
    eval_results = solaris.eval.challenges.spacenet_buildings_2(csv_pred_file, csv_gt_file)
    print("F1 Score: {}\nPrecision: {}\nRecall: {}\nTruePos: {}\nFalsePos: {}\nFalseNeg: {}".format(eval_results[1]['F1Score'].mean()*100, 
                                                                                                    eval_results[1]['Precision'].mean()*100, 
                                                                                                    eval_results[1]['Recall'].mean()*100, 
                                                                                                    eval_results[1]['TruePos'].sum(), 
                                                                                                    eval_results[1]['FalsePos'].sum(), 
                                                                                                    eval_results[1]['FalseNeg'].sum()))