import os

import bstool


if __name__ == '__main__':
    image_dir = './data/buildchange/v1/xian_fine/images'
    label_dir = './data/buildchange/v1/xian_fine/labels'

    for image_name in os.listdir(image_dir):
        file_name = bstool.get_basename(image_name)
        rgb_file = os.path.join(image_dir, image_name)
        json_file = os.path.join(label_dir, file_name + '.json')

        objects = bstool.bs_json_parse(json_file)

        masks = [obj['footprint_mask'] for obj in objects]
        bstool.show_masks_on_image(rgb_file, masks, win_name='footprint mask')

        masks = [obj['roof_mask'] for obj in objects]
        bstool.show_masks_on_image(rgb_file, masks, win_name='roof mask')