import os
import bstool
import numpy as np
import rasterio as rio
import cv2
import pandas
import glob
from shapely import affinity
import math
import mmcv


def parse_json(json_file):
    annotations = mmcv.load(json_file)['annotations']
    objects = []
    for annotation in annotations:
        object_struct = {}
        roof_mask = annotation['roof']
        roof_polygon = bstool.mask2polygon(roof_mask)
        footprint_mask = annotation['footprint']
        footprint_polygon = bstool.mask2polygon(footprint_mask)

        object_struct['roof_polygon'] = roof_polygon
        object_struct['footprint_polygon'] = footprint_polygon
        object_struct['offset'] = annotation['offset']
        object_struct['building_height'] = annotation['building_height']
                
        objects.append(object_struct)
    
    return objects


if __name__ == '__main__':
    imagesets = ['val', 'train']
    for imageset in imagesets:
        roof_csv_file = f'./data/urban3d/v0/{imageset}/urban3d_2048_roof_gt.csv'
        footprint_csv_file = f'./data/urban3d/v0/{imageset}/urban3d_2048_footprint_gt.csv'
        
        first_in = True
        min_area = 100

        generation_mode = 'labels_footprint_pixel_roof_mean'

        json_dir = f'./data/urban3d/v1/{imageset}/{generation_mode}'

        json_file_list = os.listdir(json_dir)
        for json_file in json_file_list:
            base_name = bstool.get_basename(json_file)

            json_file = os.path.join(json_dir, json_file)
            objects = parse_json(json_file)

            roof_gt_polygons, footprint_gt_polygons, gt_heights, gt_offsets = [], [], [], []
            for obj in objects:
                roof_gt_polygon = obj['roof_polygon']
                footprint_gt_polygon = obj['footprint_polygon']
                gt_height = obj['building_height']
                gt_offset = obj['offset']
                if roof_gt_polygon.area < min_area:
                    continue

                roof_gt_polygons.append(roof_gt_polygon)
                footprint_gt_polygons.append(footprint_gt_polygon)
                gt_offsets.append(gt_offset)
                gt_heights.append(gt_height)

            roof_csv_image = pandas.DataFrame({'ImageId': base_name.replace('JSON', 'RGB'),
                                            'BuildingId': range(len(roof_gt_polygons)),
                                            'PolygonWKT_Pix': roof_gt_polygons,
                                            'Confidence': 1,
                                            'Offset': gt_offsets,
                                            'Height': gt_heights})
            footprint_csv_image = pandas.DataFrame({'ImageId': base_name.replace('JSON', 'RGB'),
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