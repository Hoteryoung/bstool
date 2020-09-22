import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas
import cv2
import glob
from multiprocessing import Pool
from functools import partial
import tqdm
import mmcv

import bstool


class SplitImage():
    def __init__(self,
                 core_dataset_name='urban3d',
                 fold='ATL',
                 src_version='v1',
                 dst_version='v2',
                 imageset='val',
                 subimage_size=1024,
                 gap=512,
                 multi_processing=False,
                 num_processor=8,
                 use_weijia_labels=False):
        self.subimage_size = subimage_size
        self.gap = gap

        self.image_dir = f'./data/{core_dataset_name}/{src_version}/{imageset}/images'
        if use_weijia_labels:
            self.label_dir = f'./data/{core_dataset_name}/{src_version}/{imageset}/weijia_labels'
        else:
            self.label_dir = f'./data/{core_dataset_name}/{src_version}/{imageset}/labels'


        self.image_save_dir = f'./data/{core_dataset_name}/{dst_version}/{imageset}/images'
        bstool.mkdir_or_exist(self.image_save_dir)
        self.label_save_dir = f'./data/{core_dataset_name}/{dst_version}/{imageset}/labels'
        bstool.mkdir_or_exist(self.label_save_dir)

        self.multi_processing = multi_processing
        self.pool = Pool(num_processor)

    def parse_json(self, json_file):
        annotations = mmcv.load(json_file)['annotations']
        objects = []
        for annotation in annotations:
            object_struct = {}
            roof_mask = annotation['roof']
            roof_polygon = bstool.mask2polygon(roof_mask)
            footprint_mask = annotation['footprint']
            footprint_polygon = bstool.mask2polygon(footprint_mask)

            object_struct['polygon'] = roof_polygon
            object_struct['property'] = dict(footprint_polygon=footprint_polygon,
                                            offset=annotation['offset'],
                                            building_height=annotation['building_height'])
                    
            objects.append(object_struct)
        
        return objects

    def split_image(self, json_file):
        file_name = bstool.get_basename(json_file)

        image_file = os.path.join(self.image_dir, file_name + '.png')
        
        objects = self.parse_json(json_file)
        if len(objects) == 0:
            return

        origin_polygons = [obj['polygon'] for obj in objects]
        origin_properties = [obj['property'] for obj in objects]

        subimages = bstool.split_image(image_file, 
                                        subsize=self.subimage_size, 
                                        gap=self.gap)
        subimage_coordinates = list(subimages.keys())
        
        origin_polygons = np.array(origin_polygons)
        origin_properties = np.array(origin_properties)

        transformed_polygons = origin_polygons.copy()
        for subimage_coordinate in subimage_coordinates:
            keep = bstool.select_polygons_in_range(origin_polygons, subimage_coordinate, image_size=(self.subimage_size, self.subimage_size))
            keep_num = len(np.extract(keep == True, keep))
            if keep_num < 2:
                continue
            transformed_polygons[keep] = np.array(bstool.chang_polygon_coordinate(origin_polygons[keep].copy(), subimage_coordinate))

            subimage_properties = origin_properties[keep]
            subimage_polygons = transformed_polygons[keep]

            # bstool.show_polygons_on_image(subimages[subimage_coordinate], subimage_polygons)

            subimage_file = os.path.join(self.image_save_dir, f'{file_name}__{subimage_coordinate[0]}_{subimage_coordinate[1]}.png')
            json_file = os.path.join(self.label_save_dir, f'{file_name}__{subimage_coordinate[0]}_{subimage_coordinate[1]}.json')
            
            image_info = {"ori_filename": f"{file_name.replace('JSON', 'RGB')}.png",
                        "subimage_filename": f'{file_name}__{subimage_coordinate[0]}_{subimage_coordinate[1]}.png',
                        "width": self.subimage_size,
                        "height": self.subimage_size,
                        "coordinate": [int(_) for _ in subimage_coordinate]}

            bstool.urban3d_json_dump(subimage_polygons.tolist(), subimage_properties.tolist(), image_info, json_file)
            cv2.imwrite(subimage_file, subimages[subimage_coordinate])

    def core(self):
        json_file_list = glob.glob("{}/*.json".format(self.label_dir))
        num_image = len(json_file_list)
        if self.multi_processing:
            worker = partial(self.split_image)
            ret = list(tqdm.tqdm(self.pool.imap(worker, json_file_list), total=num_image))
            self.pool.close()
            self.pool.join()
        else:
            json_file_list = glob.glob("{}/*.json".format(self.label_dir))
            for json_file in tqdm.tqdm(json_file_list):
                self.split_image(json_file)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state) 


if __name__ == '__main__':
    core_dataset_name = 'urban3d'
    fold = 'ATL'
    src_version = 'v1'
    dst_version = 'v2'

    imagesets = ['val', 'train']
    
    subimage_size = 1024
    gap = subimage_size // 2

    for imageset in imagesets:
        print(f"Begin processing {imageset} set.")
        split_image = SplitImage(core_dataset_name=core_dataset_name,
                                fold=fold,
                                src_version=src_version,
                                dst_version=dst_version,
                                imageset=imageset,
                                subimage_size=subimage_size,
                                gap=gap,
                                multi_processing=True,
                                num_processor=8,
                                use_weijia_labels=True)
        split_image.core()
        print(f"Finish processing {imageset} set.")
