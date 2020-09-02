import os
import numpy as np
import cv2
from skimage.draw import polygon
import tqdm

import bstool


if __name__ == '__main__':
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    for city in cities:
        label_dir = f'./data/buildchange/v1/{city}/labels'
        save_dir = f'./data/buildchange/v1/{city}/pixel_offset'
        bstool.mkdir_or_exist(save_dir)

        for json_name in tqdm.tqdm(os.listdir(label_dir)):
            pixel_offset = np.zeros((1024, 1024, 2))
            file_name = bstool.get_basename(json_name)

            save_file = os.path.join(save_dir, file_name + '.npy')
            json_file = os.path.join(label_dir, file_name + '.json')

            objects = bstool.bs_json_parse(json_file)

            if len(objects) == 0:
                # generate empty offset 
                pass

            roof_masks = [obj['roof_mask'] for obj in objects]
            roof_polygons = [bstool.mask2polygon(roof_mask) for roof_mask in roof_masks]
            offsets = [obj['offset'] for obj in objects]

            for offset, roof_mask in zip(offsets, roof_masks):
                X, Y = polygon(roof_mask[0::2], roof_mask[1::2])
                
                X = np.clip(X, 0, 1023)
                Y = np.clip(Y, 0, 1023)

                pixel_offset[Y, X, 0] = int(offset[0])
                pixel_offset[Y, X, 1] = int(offset[1])

            np.save(save_file, pixel_offset.astype(np.int16))