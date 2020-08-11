import os

import bstool


if __name__ == '__main__':
    # image_dir = '/data/plane/v1/train/images'
    # anno_file = '/data/plane/v1/coco/annotations/plane_train.json'
    anno_file = '/data/urban3d/v2/coco/annotations/urban3d_v2_val_JAX_OMA.json'
    image_dir = '/data/urban3d/v2/val/images'

    coco_parser = bstool.COCOParse(anno_file)

    for image_name in os.listdir(image_dir):
        anns = coco_parser(image_name)

        image_file = os.path.join(image_dir, image_name)
        bstool.show_coco_mask(coco_parser.coco, image_file, anns)