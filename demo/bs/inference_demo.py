import numpy as np
import cv2
import geopandas
import os
import pandas
import rasterio as rio
import shapely
from shapely.geometry import Polygon, MultiPolygon
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from shapely import affinity
from collections import defaultdict
import tqdm
import ast

import mmcv
import bstool


def convert_items(result, with_offset=False, with_height=False, score_threshold=0.4, min_area=100):
    buildings = []
    if with_offset and not with_height:
        det, seg, offset = result
        height = np.zeros(offset.shape)
    if with_height:
        det, seg, offset, height = result

    bboxes = np.vstack(det)
    segms = mmcv.concat_list(seg)

    if isinstance(offset, tuple):
        offsets = offset[0]
    else:
        offsets = offset

    if with_height and isinstance(height, tuple):
        heights = height[0]
    else:
        heights = height

    for i in range(bboxes.shape[0]):
        building = dict()
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
        offset = offsets[i]
        height = heights[i][0]
        roof = mask

        roof_polygon = bstool.mask2polygon(roof)

        if roof_polygon.area < min_area:
            continue

        transform_matrix = [1, 0, 0, 1,  -1.0 * offset[0], -1.0 * offset[1]]
        footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)

        building['bbox'] = bbox.tolist()
        building['offset'] = offset.tolist()
        building['height'] = height
        building['score'] = score
        building['roof_polygon'] = roof_polygon
        building['footprint_polygon'] = footprint_polygon

        buildings.append(building)
    
    return buildings

if __name__ == '__main__':
    
    for score_threshold in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:

        img_dir = './data/buildchange/v1/other/images'
        output_dir = './data/buildchange/v1/other/vis/{}'.format(score_threshold)
        bstool.mkdir_or_exist(output_dir)
        pkl_file = '../mmdetv2-bc/results/buildchange/bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0/bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0_coco_results.pkl'
        ann_file = './data/buildchange/v1/coco/annotations/buildchange_v1_train_other.json'

        results = mmcv.load(pkl_file)

        coco = COCO(ann_file)
        img_ids = coco.get_img_ids()

        objects = defaultdict(dict)

        for idx, img_id in tqdm.tqdm(enumerate(img_ids)):
            info = coco.load_imgs([img_id])[0]
            img_name = bstool.get_basename(info['file_name'])
            img_file = os.path.join(img_dir, img_name + '.png')
            output_file = os.path.join(output_dir, img_name + '.png')

            result = results[idx]

            objects = convert_items(result, with_offset=True, score_threshold=score_threshold, min_area=0)

            footprint_masks = [bstool.polygon2mask(obj['footprint_polygon']) for obj in objects]

            img = cv2.imread(img_file)
            img = bstool.draw_masks_boundary(img, footprint_masks)
            
            # bstool.show_image(img, output_file=output_file)
            cv2.imwrite(output_file, img)