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

def get_corners(bboxes):
    """Get corners of bounding boxes
    
    Parameters
    ----------
    
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
        
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners

def bboxes_rotate(bboxes, img_shape, rotate_angle):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    assert bboxes.shape[-1] % 4 == 0
    if bboxes.shape[0] == 0:
        return bboxes
    corners = get_corners(bboxes)
    corners = np.hstack((corners, bboxes[:, 4:]))

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype = type(corners[0][0]))))
    angle = rotate_angle
    h, w, _ = img_shape
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    calculated = np.dot(M, corners.T).T
    calculated = np.array(calculated, dtype=np.float32)
    calculated = calculated.reshape(-1, 8)

    x_ = calculated[:, [0, 2, 4, 6]]
    y_ = calculated[:, [1, 3, 5, 7]]
    
    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)
    
    rotated = np.hstack((xmin, ymin, xmax, ymax))
    
    return rotated