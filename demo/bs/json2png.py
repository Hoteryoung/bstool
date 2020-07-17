import os
import numpy as np
import cv2

import bstool


if __name__ == '__main__':
    json_dir = '/data/buildchange/v1/xian_fine_origin/labels'
    output_dir = '/data/buildchange/v0/xian_fine/png_gt'
    
    for json_fn in os.listdir(json_dir):
        json_file = os.path.join(json_dir, json_fn)
        basename = bstool.get_basename(json_fn)
        objects = bstool.bs_json_parse(json_file)

        polygons = [bstool.mask2polygon(obj['roof_mask']) for obj in objects]

        foreground = bstool.generate_image(2048, 2048, (0, 0, 0))
        for idx, polygon in enumerate(polygons):
            mask = bstool.polygon2mask(polygon)
            mask = np.array(mask).reshape(1, -1, 2)
            cv2.fillPoly(foreground, mask, (255, 255, 255))

        output_file = os.path.join(output_dir, basename + '.png')
        print(output_file)
        cv2.imwrite(output_file, foreground)