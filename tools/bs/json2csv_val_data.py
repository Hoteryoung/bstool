import os
import bstool
import numpy as np
import rasterio as rio
import cv2
import pandas
import glob
from shapely import affinity


if __name__ == '__main__':
    roof_csv_file = './data/buildchange/v1/xian_fine/xian_fine_2048_roof_gt_splitted.csv'
    footprint_csv_file = './data/buildchange/v1/xian_fine/xian_fine_2048_footprint_gt_splitted.csv'
    
    first_in = True

    json_dir = './data/buildchange/v1/xian_fine/labels'

    json_file_list = glob.glob("{}/*.json".format(json_dir))
    for json_file in json_file_list:
        base_name = bstool.get_basename(json_file)

        objects = bstool.bs_json_parse(json_file)

        roof_gt_polygons = [bstool.mask2polygon(obj['roof_mask']) for obj in objects]
        footprint_gt_polygons = [bstool.mask2polygon(obj['footprint_mask']) for obj in objects]

        roof_csv_image = pandas.DataFrame({'ImageId': base_name,
                                        'BuildingId': range(len(roof_gt_polygons)),
                                        'PolygonWKT_Pix': roof_gt_polygons,
                                        'Confidence': 1})
        footprint_csv_image = pandas.DataFrame({'ImageId': base_name,
                                        'BuildingId': range(len(footprint_gt_polygons)),
                                        'PolygonWKT_Pix': footprint_gt_polygons,
                                        'Confidence': 1})
        if first_in:
            roof_csv_dataset = roof_csv_image
            footprint_csv_dataset = footprint_csv_image
            first_in = False
        else:
            roof_csv_dataset = roof_csv_dataset.append(roof_csv_image)
            footprint_csv_dataset = footprint_csv_dataset.append(footprint_csv_image)

    roof_csv_dataset.to_csv(roof_csv_file, index=False)
    footprint_csv_dataset.to_csv(footprint_csv_file, index=False)