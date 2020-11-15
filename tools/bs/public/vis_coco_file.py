import os

import bstool


if __name__ == '__main__':
    # image_dir = '/data/plane/v1/train/images'
    # anno_file = '/data/plane/v1/coco/annotations/plane_train.json'
    anno_file = '/data/buildchange/public/20201028/coco/annotations/buildchange_public_20201028_val_shanghai_xian_minarea_500.json'
    image_dir = '/data/buildchange/public/20201028/shanghai_xian/images'

    coco_parser = bstool.COCOParse(anno_file)

    for image_name in os.listdir(image_dir):
        anns = coco_parser(image_name)

        print(image_name)
        image_file = os.path.join(image_dir, image_name)
        bstool.show_coco_mask(coco_parser.coco, image_file, anns)