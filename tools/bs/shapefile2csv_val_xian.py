import os
import pandas
import glob
import math
import numpy as np

import bstool


def get_image_height_angle(objects, resolution = 0.6):
    angles = []
    for obj in objects:
        offset_x, offset_y = obj['property']['xoffset'], obj['property']['yoffset']
        if obj['property']['Floor'] is not None:
            if math.isnan(obj['property']['Floor']):
                continue
            else:
                height = obj['property']['Floor'] * 3
        else:
            continue

        angle = math.atan2(math.sqrt(offset_x ** 2 + offset_y ** 2) * resolution, height)
        angles.append(angle)
    
    if len(angles) == 0:
        height_angle = 10000
    else:
        height_angle = float(np.array(angles, dtype=np.float32).mean())

    return height_angle

def fix_invalid_height(offset, height_angle, resolution=0.6):
    offset_x, offset_y = offset

    if height_angle != 0:
        if height_angle == 10000:
            valid_height = 10000
        else:
            valid_height = (np.sqrt(offset_x ** 2 + offset_y ** 2) * resolution / np.tan(height_angle))
    else:
        valid_height = 3.0

    if valid_height > 500 or valid_height < 3:
        valid_height = 10000
    
    return valid_height


if __name__ == '__main__':
    with_fix_height = True

    sub_folds = ['arg', 'google', 'ms']

    if with_fix_height:
        roof_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt_fixed.csv'
        footprint_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_footprint_gt_fixed.csv'
    else:
        roof_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt.csv'
        footprint_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_footprint_gt.csv'

    invalid_images = ['L18_104432_210416', 'L18_104440_210384', 'L18_104440_210416', 'L18_104448_210384', 'L18_104448_210432']
    
    first_in = True
    min_area = 100

    valid_num, total_num, fixed_valid_num = 0, 0, 0

    for sub_fold in sub_folds:
        shp_dir = f'./data/buildchange/v0/xian_fine/{sub_fold}/merged_shp'
        rgb_img_dir = f'./data/buildchange/v0/xian_fine/{sub_fold}/images'

        shp_file_list = glob.glob("{}/*.shp".format(shp_dir))
        for shp_file in shp_file_list:
            base_name = bstool.get_basename(shp_file)

            if base_name in invalid_images:
                print("This xian image is invalid, skip")
                continue

            rgb_img_file = os.path.join(rgb_img_dir, base_name + '.jpg')

            objects = bstool.shp_parse(shp_file=shp_file,
                                        geo_file=rgb_img_file,
                                        src_coord='pixel',
                                        dst_coord='pixel',
                                        keep_polarity=False)

            height_angle = get_image_height_angle(objects)

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
                if obj['property']['Floor'] is not None:
                    if math.isnan(obj['property']['Floor']):
                        if with_fix_height:
                            offset = [obj['property']['xoffset'], obj['property']['yoffset']]
                            height = fix_invalid_height(offset, height_angle)
                            if height != 10000:
                                fixed_valid_num += 1
                            gt_heights.append(height)
                        else:
                            gt_heights.append(10000)
                    else:
                        gt_heights.append(obj['property']['Floor'] * 3)
                        valid_num += 1
                else:
                    if with_fix_height:
                        offset = [obj['property']['xoffset'], obj['property']['yoffset']]
                        height = fix_invalid_height(offset, height_angle)
                        if height != 10000:
                            fixed_valid_num += 1
                        gt_heights.append(height)
                    else:
                        gt_heights.append(10000)
                
                gt_properties.append(obj['property'])

                total_num += 1

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

    print(f"valid num: {valid_num}, fixed valid, num: {fixed_valid_num + valid_num}, total num: {total_num}, valid rate: {float(valid_num) / total_num}, fixed valid rate: {float(fixed_valid_num + valid_num) / total_num}")