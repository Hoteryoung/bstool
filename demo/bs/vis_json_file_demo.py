import bstool


if __name__ == '__main__':
    rgb_file = '/home/jwwangchn/Documents/100-Work/190-Intern/2020-Sensetime/codes/bstool/data/buildchange/v2/shanghai/arg/images/L18_106968_219320__0_0.png'
    json_file = '/home/jwwangchn/Documents/100-Work/190-Intern/2020-Sensetime/codes/bstool/data/buildchange/v2/shanghai/arg/labels/L18_106968_219320__0_0.json'

    objects = bstool.bs_json_parse(json_file)

    masks = [obj['footprint_mask'] for obj in objects]

    bstool.show_masks_on_image(rgb_file, masks)