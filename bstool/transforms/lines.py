import numpy as np
import itertools
import math

import bstool


def mask2wireframe(masks):
    ret_junctions = []              # N * 2
    ret_edges_positive = []         # M * 4
    ret_edges_negative = []         # K * 4

    point_number_count = 0
    for mask in masks:
        mask_point_num = len(mask) // 2
        mask_junctions = [mask[i:i + 2] for i in range(0, len(mask), 2)]
        point_combinnations = list(itertools.combinations(range(point_number_count, point_number_count + mask_point_num, 1), 2))
        
        edges_positive_coms, edges_negative_coms = [], []
        for combination in point_combinnations:
            if abs(combination[0] - combination[1]) == 1 or abs(combination[0] - combination[1]) == mask_point_num - 1:
                edges_positive_coms.append(combination)
            else:
                edges_negative_coms.append(combination)

        ret_junctions += mask_junctions
        ret_edges_positive += edges_positive_coms
        ret_edges_negative += edges_negative_coms

        point_number_count += mask_point_num

    return ret_junctions, ret_edges_positive, ret_edges_negative

def mask2lines(mask):
    mask_point_num = len(mask) // 2
    mask_junctions = [mask[i:i + 2] for i in range(0, len(mask), 2)]
    point_combinnations = list(itertools.combinations(range(0, mask_point_num, 1), 2))
    
    lines = []
    for combination in point_combinnations:
        if abs(combination[0] - combination[1]) == 1 or abs(combination[0] - combination[1]) == mask_point_num - 1:
            lines.append(mask_junctions[combination[0]] + mask_junctions[combination[1]])

    return lines

def line_angle(line, mode='atan'):
    x1, y1, x2, y2 = line

    if mode == 'atan':
        angle = math.atan2(y2 - y1, x2 - x1)
        angle = (angle + math.pi) % math.pi 
    elif mode == 'opencv':
        angle = math.atan2(y2 - y1, x2 - x1)
        angle = (angle + math.pi) % math.pi 
        
        if angle > math.pi / 2.0:
            angle = -(angle - math.pi / 2.0)
        else:
            angle = -angle
    elif mode == 'normal':
        angle = -math.atan2(x2 - x1, y2 - y1)
        angle = (angle + math.pi) % math.pi 

    return angle

def line2thetaobb(line, angle_mode='atan'):
    x1, y1, x2, y2 = line
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w = 2
    h = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    angle = line_angle(line, mode=angle_mode)

    thetaobb = [cx, cy, h, w, angle]

    return thetaobb

def line2pointobb(line, angle_mode='atan'):
    x1, y1, x2, y2 = line
    pointobb = [x1, y1, x1, y1, x2, y2, x2, y2]

    return pointobb