import os
import bstool
import numpy as np
import rasterio as rio
import cv2
import pandas
import glob
from shapely import affinity


if __name__ == '__main__':
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_folds = {'beijing':  ['arg', 'google', 'ms', 'tdt'],
                 'chengdu':  ['arg', 'google', 'ms', 'tdt'],
                 'haerbin':  ['arg', 'google', 'ms'],
                 'jinan':    ['arg', 'google', 'ms', 'tdt'],
                 'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}
    min_area = 100
    for city in cities:
        first_in = True
        roof_csv_file = f'./data/buildchange/v0/{city}/{city}_2048_roof_gt.csv'
        footprint_csv_file = f'./data/buildchange/v0/{city}/{city}_2048_footprint_gt.csv'
        for sub_fold in sub_folds[city]:
            shp_dir = f'./data/buildchange/v0/{city}/{sub_fold}/merged_shp'
            rgb_img_dir = f'./data/buildchange/v0/{city}/{sub_fold}/images'

            shp_file_list = glob.glob("{}/*.shp".format(shp_dir))
            for shp_file in shp_file_list:
                base_name = bstool.get_basename(shp_file)

                rgb_img_file = os.path.join(rgb_img_dir, base_name + '.jpg')

                objects = bstool.shp_parse(shp_file=shp_file,
                                            geo_file=rgb_img_file,
                                            src_coord='pixel',
                                            dst_coord='pixel',
                                            keep_polarity=False)

            roof_gt_polygons, gt_properties, gt_heights, gt_offsets = [], [], [], []
            for obj in objects:
                roof_gt_polygon = obj['polygon']
                if roof_gt_polygon.area < min_area:
                    continue
                valid_flag = bstool.single_valid_polygon(roof_gt_polygon)
                if not valid_flag:
                    continue
                roof_gt_polygons.append(obj['polygon'])
                gt_offsets.append([obj['property']['xoffset'], obj['property']['yoffset']])
                if 'Floor' in obj['property'].keys():
                    gt_heights.append(obj['property']['Floor'] * 3)
                elif 'half_H' in obj['property'].keys():
                    gt_heights.append(obj['property']['half_H'])
                gt_properties.append(obj['property'])

            footprint_gt_polygons = bstool.roof2footprint(roof_gt_polygons, gt_properties)

            roof_csv_image = pandas.DataFrame({'ImageId': base_name,
                                            'BuildingId': range(len(roof_gt_polygons)),
                                            'PolygonWKT_Pix': roof_gt_polygons,
                                            'Confidence': 1,
                                            'Offset': gt_offsets,
                                            'Height': gt_heights})
            footprint_csv_image = pandas.DataFrame({'ImageId': base_name,
                                            'BuildingId': range(len(footprint_gt_polygons)),
                                            'PolygonWKT_Pix': footprint_gt_polygons,
                                            'Confidence': 1,
                                            'Offset': gt_offsets,
                                            'Height': gt_heights})
            if first_in:
                roof_csv_dataset = roof_csv_image
                footprint_csv_dataset = footprint_csv_image
                first_in = False
            else:
                roof_csv_dataset = roof_csv_dataset.append(roof_csv_image)
                footprint_csv_dataset = footprint_csv_dataset.append(footprint_csv_image)

        roof_csv_dataset.to_csv(roof_csv_file, index=False)
        footprint_csv_dataset.to_csv(footprint_csv_file, index=False)