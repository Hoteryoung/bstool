# -*- encoding: utf-8 -*-
'''
@File    :   vis_compare_anno_v2_v3.py
@Time    :   2020/12/30 22:34:38
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   对令宣生成的 V2 和 V3 两个版本的标注文件进行可视化分析
'''


import numpy as np
import cv2
import os
import bstool


if __name__ == '__main__':
    v3_anno_dir = './data/buildchange/v0/shanghai/arg/anno_v3'
    vis_dir = './data/buildchange/vis/anno_v3/shanghai/arg'

    bstool.mkdir_or_exist(vis_dir)
    
    for image_fn in os.listdir(v3_anno_dir):
        basename = bstool.get_basename(image_fn)

        shp_file = f'./data/buildchange/v0/shanghai/arg/roof_shp_4326/{basename}.shp'
        geo_file = f'./data/buildchange/v0/shanghai/arg/geo_info/{basename}.png'
        rgb_file = f'./data/buildchange/v0/shanghai/arg/images/{basename}.jpg'

        if not os.path.exists(shp_file):
            continue

        v3_anno_file = os.path.join(v3_anno_dir, image_fn)
        objects = bstool.mask_parse(v3_anno_file, (1, 3))
        v3_masks = [obj['mask'] for obj in objects]

        img = cv2.imread(rgb_file)
        objects = bstool.shp_parse(shp_file=shp_file,
                                    geo_file=geo_file,
                                    src_coord='4326',
                                    dst_coord='pixel')
        v2_masks = [obj['mask'] for obj in objects]

        # v3_img = bstool.draw_masks_boundary(img.copy(), v3_masks)
        # v2_img = bstool.draw_masks_boundary(img.copy(), v2_masks)

        empty_gap = bstool.generate_image(2048, 20)

        # img = np.hstack((v2_img, empty_gap, v3_img))

        img = bstool.draw_masks_boundary(img, v2_masks, thickness=1)
        img = bstool.draw_masks_boundary(img, v3_masks, (255, 0, 0), thickness=1)
        
        # bstool.show_image(img)
        vis_file = os.path.join(vis_dir, image_fn)
        cv2.imwrite(vis_file, img)