import os
import numpy as np
import mmcv
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import cv2
import pandas
from collections import defaultdict
from tqdm import tqdm
from shapely import affinity
import shapely
from shapely.validation import explain_validity

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

                    valid_flag = bstool.single_valid_polygon(bstool.mask2polygon(mask))
                    if not valid_flag:
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

def pkl2csv_roof(pkl_file, anno_file, csv_prefix, score_threshold=0.05):
    results = mmcv.load(pkl_file)
    
    if len(results) == 0:
        return

    coco = COCO(anno_file)
    img_ids = coco.get_img_ids()

    csv_file = csv_prefix + "_" + 'roof' + '.csv'

    first_in = True
    for idx, img_id in tqdm(enumerate(img_ids)):
        info = coco.load_imgs([img_id])[0]
        img_name = info['file_name']

        det, seg = results[idx], None

        bboxes = np.vstack(det)
        segms = mmcv.concat_list(seg)
        
        single_image_bboxes = []
        single_image_masks = []
        single_image_scores = []
        for i in range(bboxes.shape[0]):
            score = bboxes[i][4]
            if score < score_threshold:
                continue

            if isinstance(segms[i]['counts'], bytes):
                segms[i]['counts'] = segms[i]['counts'].decode()
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            gray = np.array(mask * 255, dtype=np.uint8)

            contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            
            if contours != []:
                cnt = max(contours, key = cv2.contourArea)
                if cv2.contourArea(cnt) < 5:
                    continue
                mask = np.array(cnt).reshape(1, -1).tolist()[0]
                if len(mask) < 8:
                    continue

                valid_flag = bstool.single_valid_polygon(bstool.mask2polygon(mask))
                if not valid_flag:
                    continue
            else:
                continue

            bbox = bboxes[i][0:4]
            score = bboxes[i][-1]
            mask = mask

            single_image_bboxes.append(bbox.tolist())
            single_image_masks.append(bstool.mask2polygon(mask))
            single_image_scores.append(score.tolist())

        csv_image = pandas.DataFrame({'ImageId': img_name.split('.')[0],
                                      'BuildingId': range(len(single_image_masks)),
                                      'PolygonWKT_Pix': single_image_masks,
                                      'Confidence': single_image_scores})
        if first_in:
            csv_dataset = csv_image
            first_in = False
        else:
            csv_dataset = csv_dataset.append(csv_image)

    csv_dataset.to_csv(csv_file, index=False)

def pkl2csv_roof_footprint(pkl_file, anno_file, csv_prefix, score_threshold=0.05):
    if isinstance(pkl_file, str):
        results = mmcv.load(pkl_file)
    else:
        results = pkl_file
    
    if len(results) == 0:
        return

    coco = COCO(anno_file)
    img_ids = coco.get_img_ids()

    csv_file_roof = csv_prefix + "_roof.csv"
    csv_file_footprint = csv_prefix + "_footprint.csv"

    first_in = True
    progress_bar = mmcv.ProgressBar(len(img_ids))
    for idx, img_id in enumerate(img_ids):
        info = coco.load_imgs([img_id])[0]
        img_name = info['file_name']

        if len(results[idx]) == 3:
            det, seg, offset = results[idx]
            height_flag = False
        elif len(results[idx]) == 4:
            det, seg, offset, height = results[idx]
            height_flag = True

        bboxes = np.vstack(det)
        segms = mmcv.concat_list(seg)

        if isinstance(offset, tuple):
            offsets = offset[0]
        else:
            offsets = offset

        if height_flag and isinstance(height, tuple):
            heights = height[0]
        else:
            heights = height
        
        single_image_bboxes = []
        single_image_roofs = []
        single_image_footprints = []
        single_image_scores = []
        single_image_heights = []
        for i in range(bboxes.shape[0]):
            score = bboxes[i][4]
            offset = offsets[i]

            if height_flag:
                height = heights[i][0]

            if score < score_threshold:
                continue

            if isinstance(segms[i]['counts'], bytes):
                segms[i]['counts'] = segms[i]['counts'].decode()
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            gray = np.array(mask * 255, dtype=np.uint8)

            contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            
            if contours != []:
                cnt = max(contours, key = cv2.contourArea)
                if cv2.contourArea(cnt) < 5:
                    continue
                mask = np.array(cnt).reshape(1, -1).tolist()[0]
                if len(mask) < 8:
                    continue

                valid_flag = bstool.single_valid_polygon(bstool.mask2polygon(mask))
                if not valid_flag:
                    continue
            else:
                continue

            bbox = bboxes[i][0:4]
            score = bboxes[i][-1]
            roof = mask

            roof_polygon = bstool.mask2polygon(roof)

            transform_matrix = [1, 0, 0, 1,  -1.0 * offset[0], -1.0 * offset[1]]
            footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)

            single_image_bboxes.append(bbox.tolist())
            single_image_roofs.append(roof_polygon)
            single_image_footprints.append(footprint_polygon)
            single_image_scores.append(score.tolist())
            single_image_heights.append(height)

        csv_image_roof = pandas.DataFrame({'ImageId': img_name.split('.')[0],
                                      'BuildingId': range(len(single_image_roofs)),
                                      'PolygonWKT_Pix': single_image_roofs,
                                      'Confidence': single_image_scores,
                                      'BuildingHeight': single_image_heights})
        csv_image_footprint = pandas.DataFrame({'ImageId': img_name.split('.')[0],
                                      'BuildingId': range(len(single_image_footprints)),
                                      'PolygonWKT_Pix': single_image_footprints,
                                      'Confidence': single_image_scores,
                                      'BuildingHeight': single_image_heights})
        if first_in:
            csv_dataset_roof = csv_image_roof
            csv_dataset_footprint = csv_image_footprint
            first_in = False
        else:
            csv_dataset_roof = csv_dataset_roof.append(csv_image_roof)
            csv_dataset_footprint = csv_dataset_footprint.append(csv_image_footprint)

        progress_bar.update()

    csv_dataset_roof.to_csv(csv_file_roof, index=False)
    csv_dataset_footprint.to_csv(csv_file_footprint, index=False)

def merge_csv_results(input_csv_file, output_csv_file, iou_threshold=0.1, score_threshold=0.4, min_area=100):
    csv_df = pandas.read_csv(input_csv_file)
    image_name_list = list(set(csv_df.ImageId.unique()))
    
    ori_image_name_set = set()
    merged_masks = defaultdict(dict)
    merged_scores = defaultdict(dict)
    print("Indexing results of original images")
    progress_bar = mmcv.ProgressBar(len(image_name_list))
    for idx, image_name in enumerate(image_name_list):
        single_image_masks = []
        single_image_scores = []

        sub_fold, ori_image_name, coord = bstool.get_info_splitted_imagename(image_name)
        ori_image_name_set.add(ori_image_name)
        for idx, row in csv_df[csv_df.ImageId == image_name].iterrows():
            polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
            if polygon.area < min_area:
                continue
            if not bstool.single_valid_polygon(polygon):
                continue
            score = float(row.Confidence)
            if score < score_threshold:
                continue

            mask = bstool.polygon2mask(polygon)
            single_image_masks.append(mask)
            single_image_scores.append(score)

        if len(single_image_masks) == 0:
            continue
    
        merged_masks[ori_image_name][coord] = np.array(single_image_masks)
        merged_scores[ori_image_name][coord] = np.array(single_image_scores)

        progress_bar.update()

    ori_image_name_list = list(ori_image_name_set)
    print("Nms for original images")
    first_in = True
    for ori_image_name in tqdm(ori_image_name_list):
        ori_image_masks = merged_masks[ori_image_name]
        ori_image_scores = merged_scores[ori_image_name]

        nmsed_masks, nmsed_scores = bstool.merge_masks_on_subimage((ori_image_masks, ori_image_scores), 
                                                                    iou_threshold=iou_threshold)
        if len(nmsed_masks) == 0:
            continue
        nmsed_masks = [bstool.mask2polygon(mask) for mask in nmsed_masks]

        csv_image = pandas.DataFrame({'ImageId': ori_image_name,
                                      'BuildingId': range(len(nmsed_masks)),
                                      'PolygonWKT_Pix': nmsed_masks,
                                      'Confidence': nmsed_scores})
        if first_in:
            csv_dataset = csv_image
            first_in = False
        else:
            csv_dataset = csv_dataset.append(csv_image)

    csv_dataset.to_csv(output_csv_file, index=False)

def merge_masks_on_subimage(results_with_coordinate, iou_threshold=0.1):
    masks_with_coordinate, scores_with_coordinate = results_with_coordinate
    subimage_coordinates = list(masks_with_coordinate.keys())

    masks_merged, scores_merged = [], []
    keep = []
    for subimage_coordinate in subimage_coordinates:
        masks_single_image = masks_with_coordinate[subimage_coordinate]
        scores_single_image = scores_with_coordinate[subimage_coordinate]

        if len(masks_single_image) == 0:
            continue

        masks_single_image = bstool.chang_mask_coordinate(masks_single_image, subimage_coordinate)

        masks, scores = [], []
        for mask_, score_ in zip(masks_single_image, scores_single_image.tolist()):
            polygons_ = bstool.mask2polygon(mask_)
            if not bstool.single_valid_polygon(polygons_):
                continue
            masks.append(mask_)
            scores.append(score_)

        masks_merged += masks
        scores_merged += scores

        keep = bstool.mask_nms(masks_merged, np.array(scores_merged), iou_threshold=iou_threshold)

    return np.array(masks_merged)[keep].tolist(), np.array(scores_merged)[keep].tolist()

def merge_csv_results_with_height(input_csv_file, output_csv_file, iou_threshold=0.1, score_threshold=0.4, min_area=100):
    csv_df = pandas.read_csv(input_csv_file)
    image_name_list = list(set(csv_df.ImageId.unique()))
    
    ori_image_name_set = set()
    merged_masks = defaultdict(dict)
    merged_scores = defaultdict(dict)
    merged_heights = defaultdict(dict)
    print("Indexing results of original images")
    progress_bar = mmcv.ProgressBar(len(image_name_list))
    for idx, image_name in enumerate(image_name_list):
        single_image_masks = []
        single_image_scores = []
        single_image_heights = []

        sub_fold, ori_image_name, coord = bstool.get_info_splitted_imagename(image_name)
        ori_image_name_set.add(ori_image_name)
        for idx, row in csv_df[csv_df.ImageId == image_name].iterrows():
            polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
            if polygon.area < min_area:
                continue
            if not bstool.single_valid_polygon(polygon):
                continue
            score = float(row.Confidence)
            height = float(row.BuildingHeight)
            if score < score_threshold:
                continue

            mask = bstool.polygon2mask(polygon)
            single_image_masks.append(mask)
            single_image_scores.append(score)
            single_image_heights.append(height)

        if len(single_image_masks) == 0:
            continue
    
        merged_masks[ori_image_name][coord] = np.array(single_image_masks)
        merged_scores[ori_image_name][coord] = np.array(single_image_scores)
        merged_heights[ori_image_name][coord] = np.array(single_image_heights)

        progress_bar.update()

    ori_image_name_list = list(ori_image_name_set)
    print("NMS for original images")
    first_in = True
    for ori_image_name in tqdm(ori_image_name_list):
        ori_image_masks = merged_masks[ori_image_name]
        ori_image_scores = merged_scores[ori_image_name]
        ori_image_heights = merged_heights[ori_image_name]

        nmsed_masks, nmsed_scores, nmsed_heights = merge_masks_on_subimage_with_height((ori_image_masks, ori_image_scores, ori_image_heights), iou_threshold=iou_threshold)
        
        if len(nmsed_masks) == 0:
            continue
        nmsed_masks = [bstool.mask2polygon(mask) for mask in nmsed_masks]

        csv_image = pandas.DataFrame({'ImageId': ori_image_name,
                                      'BuildingId': range(len(nmsed_masks)),
                                      'PolygonWKT_Pix': nmsed_masks,
                                      'Confidence': nmsed_scores,
                                      'BuildingHeight': nmsed_heights}
                                      )
        if first_in:
            csv_dataset = csv_image
            first_in = False
        else:
            csv_dataset = csv_dataset.append(csv_image)

    csv_dataset.to_csv(output_csv_file, index=False)

def merge_masks_on_subimage_with_height(results_with_coordinate, iou_threshold=0.1):
    masks_with_coordinate, scores_with_coordinate, heights_with_coordinate = results_with_coordinate
    subimage_coordinates = list(masks_with_coordinate.keys())

    masks_merged, scores_merged, heights_merged = [], [], []
    keep = []
    for subimage_coordinate in subimage_coordinates:
        masks_single_image = masks_with_coordinate[subimage_coordinate]
        scores_single_image = scores_with_coordinate[subimage_coordinate]
        heights_single_image = heights_with_coordinate[subimage_coordinate]

        if len(masks_single_image) == 0:
            continue

        masks_single_image = bstool.chang_mask_coordinate(masks_single_image, subimage_coordinate)

        masks, scores, heights = [], [], []
        for mask_, score_, height_ in zip(masks_single_image, scores_single_image.tolist(), heights_single_image.tolist()):
            polygons_ = bstool.mask2polygon(mask_)
            if not bstool.single_valid_polygon(polygons_):
                continue
            masks.append(mask_)
            scores.append(score_)
            heights.append(height_)

        masks_merged += masks
        scores_merged += scores
        heights_merged += heights

        keep = bstool.mask_nms(masks_merged, np.array(scores_merged), iou_threshold=iou_threshold)

    return np.array(masks_merged)[keep].tolist(), np.array(scores_merged)[keep].tolist(), np.array(heights_merged)[keep].tolist()