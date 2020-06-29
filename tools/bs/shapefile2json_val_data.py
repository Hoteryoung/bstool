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


class ConvertData():
    def __init__(self,
                 core_dataset_name='buildchange',
                 src_version='v0',
                 dst_version='v1',
                 city='xian_fine',
                 sub_fold=None,
                 multi_processing=False,
                 num_processor=8):
        self.city = city
        self.sub_fold = sub_fold

        self.image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/images'
        self.roof_shp_dir = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/merged_shp'
        self.geo_info_dir = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/images'

        self.image_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}_origin/images'
        bstool.mkdir_or_exist(self.image_save_dir)
        self.label_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}_origin/labels'
        bstool.mkdir_or_exist(self.label_save_dir)

        self.multi_processing = multi_processing
        self.pool = Pool(num_processor)

    def convert(self, shapefile):
        file_name = bstool.get_basename(shapefile)

        image_file = os.path.join(self.image_dir, file_name + '.jpg')
        geo_file = os.path.join(self.geo_info_dir, file_name + '.jpg')

        img = cv2.imread(image_file)
        
        objects = bstool.shp_parse(shapefile, 
                                   geo_file, 
                                   src_coord='pixel',
                                   dst_coord='pixel',
                                   keep_polarity=False,
                                   clean_polygon_flag=True)

        origin_polygons = [obj['polygon'] for obj in objects]
        origin_properties = [obj['property'] for obj in objects]

        convert_properties = []
        for single_property in origin_properties:
            if ('xoffset' not in single_property) or ('yoffset' not in single_property):
                raise(RuntimeError("don't have xoffset and yoffset properties"))

            if "ignore" not in single_property:
                single_property['ignore'] = 0

            convert_properties.append(single_property)

        origin_properties = convert_properties

        image_file = os.path.join(self.image_save_dir, f'{file_name}.png')
        json_file = os.path.join(self.label_save_dir, f'{file_name}.json')
        
        image_info = {"ori_filename": f"{file_name}.jpg",
                    "subimage_filename": f'{file_name}.png',
                    "width": img.shape[0],
                    "height": img.shape[1],
                    "city": self.city,
                    "sub_fold": self.sub_fold,
                    "coordinate": [int(_) for _ in (0, 0)]}

        bstool.bs_json_dump(origin_polygons, origin_properties, image_info, json_file)
        cv2.imwrite(image_file, img)

    def core(self):
        shp_file_list = glob.glob("{}/*.shp".format(self.roof_shp_dir))
        num_image = len(shp_file_list)
        if self.multi_processing:
            worker = partial(self.convert)
            ret = list(tqdm.tqdm(self.pool.imap(worker, shp_file_list), total=num_image))
            self.pool.close()
            self.pool.join()
        else:
            shp_file_list = glob.glob("{}/*.shp".format(self.roof_shp_dir))
            for shp_file in tqdm.tqdm(shp_file_list):
                self.convert(shp_file)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state) 


if __name__ == '__main__':
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    dst_version = 'v1'
    
    cities = ['xian_fine']
    sub_folds = {'xian_fine':  ['arg', 'google', 'ms']}

    for city in cities:
        for sub_fold in sub_folds[city]:
            print("Begin processing {} {} set.".format(city, sub_fold))
            convert_data = ConvertData(core_dataset_name=core_dataset_name,
                                    src_version=src_version,
                                    dst_version=dst_version,
                                    city=city,
                                    sub_fold=sub_fold,
                                    multi_processing=True,
                                    num_processor=8)
            convert_data.core()
            print("Finish processing {} {} set.".format(city, sub_fold))
