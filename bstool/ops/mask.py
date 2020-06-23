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
    boxes = np.array([bstool.mask2bbox(_) for _ in masks])
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    order = scores.argsort()[::-1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    keep = []
    while order.size > 0:
        best_box = order[0]
        keep.append(best_box)

        inter_x1 = np.maximum(x1[order[1:]], x1[best_box])
        inter_y1 = np.maximum(y1[order[1:]], y1[best_box])
        inter_x2 = np.minimum(x2[order[1:]], x2[best_box])
        inter_y2 = np.minimum(y2[order[1:]], y2[best_box])

        inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0.0)
        inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0.0)

        inter = inter_w * inter_h

        iou = inter / (areas[best_box] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        
        order = order[inds + 1]

    return keep