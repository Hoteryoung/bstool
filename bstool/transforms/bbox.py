import numpy as np

import bstool


def xyxy2cxcywh(bbox):
    """bbox format convert

    Args:
        bbox (list): [xmin, ymin, xmax, ymax]

    Returns:
        list: [cx, cy, w, h]
    """
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    w = xmax - xmin
    h = ymax - ymin
    
    return [cx, cy, w, h]

def cxcywh2xyxy(bbox):
    """bbox format convert

    Args:
        bbox (list): [cx, cy, w, h]

    Returns:
        list: [xmin, ymin, xmax, ymax]
    """
    cx, cy, w, h = bbox
    xmin = int(cx - w / 2.0)
    ymin = int(cy - h / 2.0)
    xmax = int(cx + w / 2.0)
    ymax = int(cy + h / 2.0)
    
    return [xmin, ymin, xmax, ymax]

def xywh2xyxy(bbox):
    """bbox format convert

    Args:
        bbox (list): [xmin, ymin, w, h]

    Returns:
        list: [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, w, h = bbox
    xmax = xmin + w
    ymax = ymin + h
    
    return [xmin, ymin, xmax, ymax]

def xyxy2xywh(bbox):
    """bbox format convert

    Args:
        bbox (list): [xmin, ymin, xmax, ymax]

    Returns:
        list: [xmin, ymin, w, h]
    """
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    
    return [xmin, ymin, w, h]

def chang_bbox_coordinate(bboxes, coordinate):
    """change the coordinate of bbox

    Args:
        bboxes (np.array): [N, 4], (xmin, ymin, xmax, ymax)
        coordinate (list or tuple): distance of moving

    Returns:
        np.array: converted bboxes
    """
    bboxes[:, 0::2] = bboxes[:, 0::2] + coordinate[0]
    bboxes[:, 1::2] = bboxes[:, 1::2] + coordinate[1]

    return bboxes