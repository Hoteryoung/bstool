import bstool


if __name__ == '__main__':
    rgb_file = './data/buildchange/demo/images/L18_106968_219320__0_0.png'
    json_file = './data/buildchange/demo/labels/L18_106968_219320__0_0.json'

    objects = bstool.bs_json_parse(json_file)

    masks = [obj['footprint_mask'] for obj in objects]

    bstool.show_masks_on_image(rgb_file, masks)