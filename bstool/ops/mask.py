import numpy as np
import cv2

import bstool


def mask_nms(masks, scores, iou_threshold=0.5):
    """non-maximum suppression (NMS) on the masks according to their intersection-over-union (IoU)
    
    Arguments:
        masks {np.array} -- [N * 4]
        scores {np.array} -- [N * 1]
        iou_threshold {float} -- threshold for IoU
    """
    polygons = np.array([bstool.mask2polygon(mask) for mask in masks])

    areas = np.array([polygon.area for polygon in polygons])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        best_mask_idx = order[0]
        keep.append(best_mask_idx)

        best_mask = polygons[best_mask_idx]
        remain_masks = polygons[order[1:]]

        inters = []
        for remain_mask in remain_masks:
            mask1 = best_mask
            mask2 = remain_mask
            inter = mask1.intersection(mask2).area
            inters.append(inter)

        inters = np.array(inters)
        iou = inters / (areas[best_mask_idx] + areas[order[1:]] - inters)

        inds = np.where(iou <= iou_threshold)[0]
        
        order = order[inds + 1]

    return keep