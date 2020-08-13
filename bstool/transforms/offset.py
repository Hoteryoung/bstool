import math
import numpy as np

import bstool


def offset_coordinate_transform(offset, transform_flag='xy2la'):
    """transform the coordinate of offsets

    Args:
        offset (list): list of offset
        transform_flag (str, optional): flag of transform. Defaults to 'xy2la'.

    Raises:
        NotImplementedError: [description]

    Returns:
        list: transformed offsets
    """
    if transform_flag == 'xy2la':
        offset_x, offset_y = offset
        length = math.sqrt(offset_x ** 2 + offset_y ** 2)
        angle = math.atan2(offset_y, offset_x)
        offset = [length, angle]
    elif transform_flag == 'la2xy':
        length, angle = offset
        offset_x = length * np.cos(angle)
        offset_y = length * np.sin(angle)
        offset = [offset_x, offset_y]
    else:
        raise NotImplementedError

    return offset

def offset_flip(offsets, transform_flag='h'):
    """flip offset vector

    Args:
        offsets (list): list of offset
        transform_flag (str, optional): horizontal or vertical flip. Defaults to 'h'.

    Raises:
        NotImplementedError: other flag of transform

    return:
        list: flipped offset
    """
    if transform_flag == 'h':
        offsets = [[-offset[0], offset[1]] for offset in offsets]
    elif transform_flag == 'v':
        offsets = [[offset[0], -offset[1]] for offset in offsets]
    else: 
        raise NotImplementedError

    return offsets

def offset_rotate(offsets, angle=0):
    """rotate offset vector

    Args:
        offsets (list): list of offset
        angle (int, optional): Rotation angle in degrees, positive values mean
            anticlockwise rotation. Defaults to 0. (rad/s), 

    Returns:
        list: list of transformed offset vector
    """
    offsets = [offset_coordinate_transform(offset, transform_flag='xy2la') for offset in offsets]

    offsets = [[offset[0], offset[1] + angle] for offset in offsets]

    offsets = [offset_coordinate_transform(offset, transform_flag='la2xy') for offset in offsets]

    return offsets