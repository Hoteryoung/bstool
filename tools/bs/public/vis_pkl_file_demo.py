import os
import numpy as np
import mmcv
import bstool
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from shapely import affinity
from collections import defaultdict
import tqdm


def convert_items_offset(result, score_threshold=0.4, min_area=0):
    buildings = []
    det, seg, offset = result

    bboxes = np.vstack(det)
    segms = mmcv.concat_list(seg)

    if isinstance(offset, tuple):
        offsets = offset[0]
    else:
        offsets = offset

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
        else:
            continue

        bbox = bboxes[i][0:4]
        roof = mask
        offset = offsets[i]

        roof_polygon = bstool.mask2polygon(roof)
    
        valid_flag = bstool.single_valid_polygon(roof_polygon)
        if not valid_flag:
            roof_polygon = roof_polygon.buffer(0)
            if roof_polygon.geom_type == 'MultiPolygon':
                continue

        # if roof_polygon.area < min_area:
        #     continue

        transform_matrix = [1, 0, 0, 1,  -1.0 * offset[0], -1.0 * offset[1]]
        footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)

        building['bbox'] = bbox.tolist()
        building['score'] = score
        building['footprint_mask'] = bstool.polygon2mask(footprint_polygon)
        building['roof_mask'] = roof
        building['offset'] = offset

        buildings.append(building)
    
    return buildings

def convert_items_bbox(result, score_threshold=0.4, min_area=0):
    buildings = []
    det, seg, offset = result

    bboxes = np.vstack(det)

    for i in range(bboxes.shape[0]):
        building = dict()
        score = bboxes[i][4]
        if score < score_threshold:
            continue

        bbox = bboxes[i][0:4]

        building['bbox'] = bbox.tolist()

        buildings.append(building)
    
    return buildings


def convert_items_baseline(result, score_threshold=0.4, min_area=0):
    buildings = []
    det, seg = result[:2]

    bboxes = np.vstack(det)
    segms = mmcv.concat_list(seg)

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

            # valid_flag = bstool.single_valid_polygon(bstool.mask2polygon(mask))
            # if not valid_flag:
                # .buffer(0)
                # continue
        else:
            continue

        bbox = bboxes[i][0:4]
        footprint = mask

        building['bbox'] = bbox.tolist()
        building['score'] = score
        building['footprint_mask'] = footprint

        buildings.append(building)
    
    return buildings


if __name__ == '__main__':
    version = 'offset'
    color = (75, 25, 230)
    if version == 'baseline':
        pkl_file = './data/buildchange/results/bc_v100.01.09_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline/bc_v100.01.09_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_shanghai_xian_public_coco_results.pkl'
        ann_file = './data/buildchange/public/20201028/coco/annotations/buildchange_public_20201028_val_shanghai_xian_minarea_500.json'
        # ann_file = '/data/buildchange/buildchange_v1_val_xian_fine.json'
        image_dir = '/data/buildchange/public/20201028/shanghai_xian/images'

        vis_dir = '/data/buildchange/vis/paper/baseline'
        bstool.mkdir_or_exist(vis_dir)

        # buildings = 

        results = mmcv.load(pkl_file)
        coco = COCO(ann_file)
        img_ids = coco.get_img_ids()

        objects = dict()
        for idx, img_id in tqdm.tqdm(enumerate(img_ids)):
            
            info = coco.load_imgs([img_id])[0]
            img_name = bstool.get_basename(info['file_name'])

            if img_name != 'L18_104512_210416__0_1024':
                continue

            result = results[idx]

            buildings = convert_items_baseline(result, score_threshold=0.4)

            img = cv2.imread(os.path.join(image_dir, img_name + '.png'))

            footprint_masks = [obj['footprint_mask'] for obj in buildings]
            img = bstool.draw_masks_boundary(img, footprint_masks, color)

            vis_file = os.path.join(vis_dir, img_name + '.png')
            cv2.imwrite(vis_file, img)

            print("===================== Finish processing: ============================", img_name)
            
    if version == 'offset':
        # pkl_file = '/data/buildchange/results/bc_v100.02.08_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_without_image_rotation/bc_v100.02.08_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_without_image_rotation_shanghai_xian_public_coco_results.pkl'
        pkl_file = '/data/buildchange/results/bc_v100.02.01_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles/bc_v100.02.01_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_shanghai_xian_public_coco_results.pkl'

        ann_file = './data/buildchange/public/20201028/coco/annotations/buildchange_public_20201028_val_shanghai_xian_minarea_500.json'
        # ann_file = '/data/buildchange/buildchange_v1_val_xian_fine.json'
        image_dir = '/data/buildchange/public/20201028/shanghai_xian/images'

        vis_dir = '/data/buildchange/vis/paper/LOVE'
        bstool.mkdir_or_exist(vis_dir)

        # buildings = 

        results = mmcv.load(pkl_file)
        coco = COCO(ann_file)
        img_ids = coco.get_img_ids()

        objects = dict()
        for idx, img_id in tqdm.tqdm(enumerate(img_ids)):
            
            info = coco.load_imgs([img_id])[0]
            img_name = bstool.get_basename(info['file_name'])

            if img_name != 'L18_104512_210416__0_1024':
                continue

            result = results[idx]

            buildings = convert_items_offset(result, score_threshold=0.4)

            img = cv2.imread(os.path.join(image_dir, img_name + '.png'))

            footprint_masks = [obj['footprint_mask'] for obj in buildings]
            img = bstool.draw_masks_boundary(img, footprint_masks, color)

            vis_file = os.path.join(vis_dir, img_name + '.png')
            cv2.imwrite(vis_file, img)

            print("===================== Finish processing: ============================", img_name)

    if version == 'bbox':
        pkl_file = '/data/buildchange/results/bc_v100.02.08_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_without_image_rotation/bc_v100.02.08_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_without_image_rotation_shanghai_xian_public_coco_results.pkl'
        ann_file = './data/buildchange/public/20201028/coco/annotations/buildchange_public_20201028_val_shanghai_xian_minarea_500.json'
        # ann_file = '/data/buildchange/buildchange_v1_val_xian_fine.json'
        image_dir = '/data/buildchange/public/20201028/shanghai_xian/images'

        vis_dir = '/data/buildchange/vis/paper/LOVE'
        bstool.mkdir_or_exist(vis_dir)

        results = mmcv.load(pkl_file)
        coco = COCO(ann_file)
        img_ids = coco.get_img_ids()

        objects = dict()
        for idx, img_id in tqdm.tqdm(enumerate(img_ids)):
            
            info = coco.load_imgs([img_id])[0]
            img_name = bstool.get_basename(info['file_name'])

            if img_name != 'L18_104400_210392__1024_0':
                continue

            result = results[idx]

            buildings = convert_items_bbox(result, score_threshold=0.4)

            img = cv2.imread(os.path.join(image_dir, img_name + '.png'))

            for obj in buildings:
                bbox = obj['bbox']
                bbox = bstool.xyxy2xywh(bbox)
                img = bstool.show_bbox(img, bbox)

            vis_file = os.path.join(vis_dir, img_name + '.png')
            cv2.imwrite(vis_file, img)

            print("===================== Finish processing: ============================", img_name)