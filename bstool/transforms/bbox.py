import numpy as np


def xyxy2cxcywh(bbox):
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    w = xmax - xmin
    h = ymax - ymin
    
    return [cx, cy, w, h]

def cxcywh2xyxy(bbox):
    cx, cy, w, h = bbox
    xmin = int(cx - w / 2.0)
    ymin = int(cy - h / 2.0)
    xmax = int(cx + w / 2.0)
    ymax = int(cy + h / 2.0)
    
    return [xmin, ymin, xmax, ymax]

def xywh2xyxy(bbox):
    xmin, ymin, w, h = bbox
    xmax = xmin + w
    ymax = ymin + h
    
    return [xmin, ymin, xmax, ymax]

def xyxy2xywh(bbox):
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    
    return [xmin, ymin, w, h]