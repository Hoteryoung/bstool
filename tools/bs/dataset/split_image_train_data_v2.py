import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas
import cv2
import glob
from multiprocessing import Pool
from functools import partial
import tqdm

import bstool


class SplitImage():
    def __init__(self,
                 core_dataset_name='buildchange',
                 src_version='v0',
                 dst_version='v1',
                 city='shanghai',
                 sub_fold=None,
                 subimage_size=1024,
                 gap=512,
                 multi_processing=False,
                 num_processor=8):
        self.city = city
        self.sub_fold = sub_fold
        self.subimage_size = subimage_size
        self.gap = gap

        self.image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/images'
        if city == 'chengdu':
            self.json_dir = f'/mnt/lustre/menglingxuan/buildingwolf/20200329/{city}_L18/{sub_fold}/anno_20200924/OffsetField/TXT'
        else:
            self.json_dir = f'/mnt/lustre/menglingxuan/buildingwolf/20200329/{city}_18/{sub_fold}/anno_20200924/OffsetField/TXT'
            # self.json_dir = '/data/buildchange/v0/shanghai/arg/json_20200924'

        self.image_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}/images'
        bstool.mkdir_or_exist(self.image_save_dir)
        self.label_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}/labels'
        bstool.mkdir_or_exist(self.label_save_dir)

        self.multi_processing = multi_processing
        self.pool = Pool(num_processor)

    def split_image(self, json_file):
        file_name = bstool.get_basename(json_file)

        image_file = os.path.join(self.image_dir, file_name + '.jpg')
        
        objects = bstool.lingxuan_json_parser(json_file)

        if len(objects) == 0:
            return

        # footprint polygons
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

            # clip the polygon on boundary (has some bugs)
            # transformed_polygons[keep] = np.array(bstool.clip_boundary_polygon(transformed_polygons[keep], image_size=(self.subimage_size, self.subimage_size)))

            drop = bstool.drop_subimage(subimages, subimage_coordinate, transformed_polygons[keep])

            if drop:
                continue

            subimage_properties = origin_properties[keep]
            subimage_polygons = transformed_polygons[keep]

            # bstool.show_polygons_on_image(subimages[subimage_coordinate], subimage_polygons)

            subimage_file = os.path.join(self.image_save_dir, f'{self.city}_{self.sub_fold}__{file_name}__{subimage_coordinate[0]}_{subimage_coordinate[1]}.png')
            json_file = os.path.join(self.label_save_dir, f'{self.city}_{self.sub_fold}__{file_name}__{subimage_coordinate[0]}_{subimage_coordinate[1]}.json')
            
            image_info = {"ori_filename": f"{file_name}.jpg",
                        "subimage_filename": f'{self.city}_{self.sub_fold}__{file_name}__{subimage_coordinate[0]}_{subimage_coordinate[1]}.png',
                        "width": self.subimage_size,
                        "height": self.subimage_size,
                        "city": self.city,
                        "sub_fold": self.sub_fold,
                        "coordinate": [int(_) for _ in subimage_coordinate]}

            bstool.bs_json_dump_v2(subimage_polygons.tolist(), subimage_properties.tolist(), image_info, json_file)
            cv2.imwrite(subimage_file, subimages[subimage_coordinate])

    def core(self):
        json_file_list = glob.glob("{}/*.json".format(self.json_dir))
        num_image = len(json_file_list)
        if self.multi_processing:
            worker = partial(self.split_image)
            ret = list(tqdm.tqdm(self.pool.imap(worker, json_file_list), total=num_image))
            self.pool.close()
            self.pool.join()
        else:
            json_file_list = glob.glob("{}/*.json".format(self.json_dir))
            for json_file in tqdm.tqdm(json_file_list):
                self.split_image(json_file)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state) 


if __name__ == '__main__':
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    dst_version = 'v2'

    # cities = ['shanghai']
    # sub_folds = {'shanghai': ['arg']}

    # cities = ['beijing', 'jinan', 'haerbin', 'chengdu']                 # debug
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_folds = {'beijing':  ['arg', 'google', 'ms', 'tdt'],
                 'chengdu':  ['arg', 'google', 'ms', 'tdt'],
                 'haerbin':  ['arg', 'google', 'ms'],
                 'jinan':    ['arg', 'google', 'ms', 'tdt'],
                 'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}
    
    subimage_size = 1024
    gap = subimage_size // 2

    for city in cities:
        for sub_fold in sub_folds[city]:
            print("Begin processing {} {} set.".format(city, sub_fold))
            split_image = SplitImage(core_dataset_name=core_dataset_name,
                                    src_version=src_version,
                                    dst_version=dst_version,
                                    city=city,
                                    sub_fold=sub_fold,
                                    subimage_size=subimage_size,
                                    gap=gap,
                                    multi_processing=True,
                                    num_processor=8)
            split_image.core()
            print("Finish processing {} {} set.".format(city, sub_fold))
