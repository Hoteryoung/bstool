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
        self.roof_shp_dir = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/roof_shp_4326'
        self.geo_info_dir = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/geo_info'
        self.ignore_info_dir = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/pixel_anno_v2'

        wrong_file = f'./data/{core_dataset_name}/{src_version}/{city}/{sub_fold}/wrongShpFile.txt'
        self.skip_filenames = self.read_wrong_file(wrong_file)

        self.image_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}/images'
        bstool.mkdir_or_exist(self.image_save_dir)
        self.label_save_dir = f'./data/{core_dataset_name}/{dst_version}/{city}/labels'
        bstool.mkdir_or_exist(self.label_save_dir)

        self.multi_processing = multi_processing
        self.pool = Pool(num_processor)
    
    def read_wrong_file(self, wrong_file):
        skip_filenames = []
        with open(wrong_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                skip_filename = bstool.get_basename(line.strip('\n'))
                skip_filenames.append(skip_filename)
        
        return skip_filenames

    def split_image(self, shapefile):
        file_name = bstool.get_basename(shapefile)
        if file_name in self.skip_filenames:
            return

        image_file = os.path.join(self.image_dir, file_name + '.jpg')
        geo_file = os.path.join(self.geo_info_dir, file_name + '.png')
        ignore_file = os.path.join(self.ignore_info_dir, file_name + '.png')
        
        objects = bstool.shp_parse(shapefile, 
                                   geo_file, 
                                   src_coord='4326',
                                   dst_coord='pixel',
                                   clean_polygon_flag=True)
        if len(objects) == 0:
            return

        origin_polygons = [obj['polygon'] for obj in objects]
        origin_properties = [obj['property'] for obj in objects]

        objects = bstool.mask_parse(ignore_file, 
                                    subclasses=255,
                                    clean_polygon_flag=True)
        if len(objects) > 0:
            ignore_polygons = [obj['polygon'] for obj in objects]

            ignored_polygon_indexes = bstool.get_ignored_polygon_idx(origin_polygons, ignore_polygons)

            origin_properties = bstool.add_ignore_flag_in_property(origin_properties, ignored_polygon_indexes)

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
            polygons_ = transformed_polygons[keep]
            polygons_ = bstool.clip_boundary_polygon(polygons_, image_size=(self.subimage_size, self.subimage_size))
            print(type(polygons_))
            transformed_polygons[keep] = np.array(polygons_)
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

            bstool.bs_json_dump(subimage_polygons.tolist(), subimage_properties.tolist(), image_info, json_file)
            cv2.imwrite(subimage_file, subimages[subimage_coordinate])

    def core(self):
        shp_file_list = glob.glob("{}/*.shp".format(self.roof_shp_dir))
        num_image = len(shp_file_list)
        if self.multi_processing:
            worker = partial(self.split_image)
            ret = list(tqdm.tqdm(self.pool.imap(worker, shp_file_list), total=num_image))
            self.pool.close()
            self.pool.join()
        else:
            shp_file_list = glob.glob("{}/*.shp".format(self.roof_shp_dir))
            for shp_file in tqdm.tqdm(shp_file_list):
                self.split_image(shp_file)

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

    cities = ['shanghai']
    sub_folds = {'shanghai': ['PHR2017']}

    # cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    # sub_folds = {'beijing':  ['arg', 'google', 'ms', 'tdt'],
    #              'chengdu':  ['arg', 'google', 'ms', 'tdt'],
    #              'haerbin':  ['arg', 'google', 'ms'],
    #              'jinan':    ['arg', 'google', 'ms', 'tdt'],
    #              'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}
    
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
