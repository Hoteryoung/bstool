import cv2
import os
import bstool
import os

import bstool


if __name__ == '__main__':
    image_dir = './data/urban3d/v2/val/images'
    label_dir = './data/urban3d/v2/val/labels'

    for image_name in os.listdir(image_dir):
        file_name = bstool.get_basename(image_name)
        rgb_file = os.path.join(image_dir, image_name)
        json_file = os.path.join(label_dir, file_name + '.json')

        objects = bstool.urban3d_json_parse(json_file)

        if len(objects) == 0:
            continue

        masks = [obj['footprint_mask'] for obj in objects]
        bboxes = [obj['footprint_bbox'] for obj in objects]
        bboxes = [bstool.xywh2xyxy(bbox) for bbox in bboxes]
        bstool.show_bboxs_on_image(rgb_file, bboxes, win_name='footprint bbox')
        bstool.show_masks_on_image(rgb_file, masks, win_name='footprint mask')

        masks = [obj['roof_mask'] for obj in objects]
        bstool.show_masks_on_image(rgb_file, masks, win_name='roof mask')