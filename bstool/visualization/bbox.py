import numpy as np
import cv2

import bstool


def show_bboxs_on_image(img,
                        bboxes,
                        labels=None,
                        scores=None,
                        score_threshold=0.0,
                        show_label=False,
                        show_score=False,
                        thickness=2,
                        show=True,
                        win_name='',
                        wait_time=0,
                        output_file=None,
                        return_img=False):
    """ Draw horizontal bounding boxes on image

    """
    if isinstance(img, str):
        img = cv2.imread(img)
        
    if len(bboxes) == 0:
        return

    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)

    if bboxes.ndim == 1:
        bboxes = np.array([bboxes])

    if labels is None:
        labels_vis = np.array(['ins'] * bboxes.shape[0])
    else:
        labels_vis = [list(label.keys())[0] for label in labels]

    color_list = list(bstool.COLORS.keys())

    if scores is None:
        scores_vis = np.array([1.0] * bboxes.shape[0])
    else:
        scores_vis = np.array(scores)
        if scores_vis.ndim == 0:
            scores_vis = np.array([scores_vis])

    for idx, (bbox, label, score) in enumerate(zip(bboxes, labels_vis, scores_vis)):
        if score < score_threshold:
            continue
        bbox = bbox.astype(np.int32)
        xmin, ymin, xmax, ymax = bbox

        current_color = (bstool.COLORS[color_list[idx % 20]][2], bstool.COLORS[color_list[idx % 20]][1], bstool.COLORS[color_list[idx % 20]][0])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=current_color, thickness=thickness)
        
        if show_label:
            cv2.putText(img, label, (xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = current_color, thickness = 2, lineType = 8)
        if show_score:
            cv2.putText(img, "{:.2f}".format(score), (xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = current_color, thickness = 2, lineType = 8)
    if show:
        bstool.show_image(img, output_file=output_file, win_name=win_name, wait_time=wait_time)

    if return_img:
        return img