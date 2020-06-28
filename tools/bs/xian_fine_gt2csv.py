import os
import bstool
import numpy as np
import rasterio as rio
import cv2
import pandas
import glob
from shapely import affinity


if __name__ == '__main__':
    sub_folds = ['arg', 'google', 'ms']

    roof_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt.csv'
    footprint_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_footprint_gt.csv'
    
    first_in = True

    for sub_fold in sub_folds:
        shp_dir = f'./data/buildchange/v0/xian_fine/{sub_fold}/merged_shp'
        rgb_img_dir = f'./data/buildchange/v0/xian_fine/{sub_fold}/images'

        shp_file_list = glob.glob("{}/*.shp".format(shp_dir))
        for shp_file in shp_file_list:
            base_name = bstool.get_basename(shp_file)

            rgb_img_file = os.path.join(rgb_img_dir, base_name + '.jpg')

            objects = bstool.shp_parse(shp_file=shp_file,
                                        geo_file=rgb_img_file,
                                        src_coord='pixel',
                                        dst_coord='pixel',
                                        keep_polarity=False)

            roof_gt_polygons = [obj['polygon'] for obj in objects]
            gt_properties = [obj['property'] for obj in objects]

            footprint_gt_polygons = bstool.roof2footprint(roof_gt_polygons, gt_properties)

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