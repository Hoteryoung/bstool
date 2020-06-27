import numpy as np

import bstool


def merge_results_on_subimage(results_with_coordinate, iou_threshold=0.5):
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

        bboxes_single_image = bstool.chang_bbox_coordinate(bboxes_single_image, subimage_coordinate)
        masks_single_image = bstool.chang_mask_coordinate(masks_single_image, subimage_coordinate)

        bboxes_merged += bboxes_single_image.tolist()
        masks_merged += masks_single_image
        scores_merged += scores_single_image.tolist()

    keep = bstool.bbox_nms(np.array(bboxes_merged), np.array(scores_merged), iou_threshold=iou_threshold)

    return np.array(bboxes_merged)[keep].tolist(), np.array(masks_merged)[keep]