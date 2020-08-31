import os
import cv2
import math
import numpy as np

import bstool


if __name__ == '__main__':
    image_dir = './data/buildchange/v1/xian_fine/images'
    label_dir = './data/buildchange/v1/xian_fine/labels'
    
    with_flip = False
    with_rotate = True

    for image_name in os.listdir(image_dir):
        file_name = bstool.get_basename(image_name)
        rgb_file = os.path.join(image_dir, image_name)
        json_file = os.path.join(label_dir, file_name + '.json')

        img = cv2.imread(rgb_file)
        objects = bstool.bs_json_parse(json_file)

        if len(objects) == 0:
            continue

        offsets = [obj['offset'] for obj in objects]
        roof_masks = [obj['roof_mask'] for obj in objects]
        footprint_masks = [bstool.polygon2mask(bstool.roof2footprint_single(bstool.mask2polygon(roof_mask), offset, offset_model='footprint2roof')) for roof_mask, offset in zip(roof_masks, offsets)]

        if with_flip:
            img = bstool.image_flip(img, transform_flag='v')
            offsets = bstool.offset_flip(offsets, transform_flag='v')
            roof_masks = bstool.mask_flip(roof_masks, transform_flag='v', image_size=(1024, 1024))
            footprint_masks = [bstool.polygon2mask(bstool.roof2footprint_single(bstool.mask2polygon(roof_mask), offset, offset_model='footprint2roof')) for roof_mask, offset in zip(roof_masks, offsets)]
        
        if with_rotate:
            angle = np.random.choice([270, 290, 300, 350, 259]) * np.pi / 180.0
            img = bstool.image_rotate(img, angle=angle)
            offsets = bstool.offset_rotate(offsets, angle=angle)
            roof_masks = bstool.mask_rotate(roof_masks, angle=angle)
            footprint_masks = [bstool.polygon2mask(bstool.roof2footprint_single(bstool.mask2polygon(roof_mask), offset, offset_model='footprint2roof')) for roof_mask, offset in zip(roof_masks, offsets)]

        bstool.show_masks_on_image(img, roof_masks, win_name='roof mask')
        bstool.show_masks_on_image(img, footprint_masks, win_name='footprint mask')
        
        
        