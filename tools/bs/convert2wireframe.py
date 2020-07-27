import os
import numpy as np
import mmcv
import json

import bstool


if __name__ == '__main__':
    image_dir = './data/buildchange/v1/jinan/images'
    label_dir = './data/buildchange/v1/jinan/labels'
    json_file = './data/buildchange/v1/jinan/wireframe/train.json'

    bstool.mkdir_or_exist('./data/buildchange/v1/jinan/wireframe/')

    height, width = 1024, 1024

    json_data = []
    progress_bar = mmcv.ProgressBar(len(os.listdir(label_dir)))
    for label_fn in os.listdir(label_dir):
        image_anns = {}
        label_file = os.path.join(label_dir, label_fn)

        ori_anns = mmcv.load(label_file)['annotations']

        image_anns['width'] = width
        image_anns['height'] = height

        junctions, edges_positive, edges_negative = [], [], []
        masks = []
        for ori_ann in ori_anns:
            mask = ori_ann['roof']
            masks.append(mask)
            
        wireframe_items = bstool.mask2wireframe(masks)
        junctions += wireframe_items[0]
        edges_positive += wireframe_items[1]
        edges_negative += wireframe_items[2]

        # bstool.show_masks_on_image(bstool.generate_image(1024, 1024), masks)
        
        image_anns['junctions'] = junctions
        image_anns['edges_positive'] = edges_positive
        image_anns['edges_negative'] = edges_negative
        image_anns['filename'] = bstool.get_basename(label_fn) + '.png'

        json_data.append(image_anns)

        progress_bar.update()

    with open(json_file, "w") as jsonfile:
        json.dump(json_data, jsonfile, indent = 4)