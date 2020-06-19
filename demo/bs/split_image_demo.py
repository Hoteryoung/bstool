import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas
import cv2

import bstool


if __name__ == '__main__':
    file_name = 'L18_106968_219320'
    shp_file = './data/buildchange/v0/shanghai/merged_shp/{}.shp'.format(file_name)
    ignore_file = './data/buildchange/v0/shanghai/anno_v2/{}.png'.format(file_name)
    geo_file = './data/buildchange/v0/shanghai/geo_info/{}.png'.format(file_name)
    rgb_file = './data/buildchange/v0/shanghai/images/{}.jpg'.format(file_name)

    json_save_dir = './data/buildchange/demo/labels'
    image_save_dir = './data/buildchange/demo/images'

    bstool.mkdir_or_exist(json_save_dir)
    bstool.mkdir_or_exist(image_save_dir)

    objects = bstool.shp_parse(shp_file=shp_file,
                                geo_file=geo_file,
                                src_coord='4326',
                                dst_coord='pixel')
    origin_polygons = [obj['polygon'] for obj in objects]
    origin_properties = [obj['property'] for obj in objects]

    objects = bstool.mask_parse(ignore_file, subclasses=255)
    ignore_polygons = [obj['polygon'] for obj in objects]

    ignored_polygon_indexes = bstool.get_ignored_polygon_idx(origin_polygons, ignore_polygons)

    origin_properties = bstool.add_ignore_flag_in_property(origin_properties, ignored_polygon_indexes)

    subsize = 1024
    subimages = bstool.split_image(rgb_file, subsize=subsize, gap=512)
    subimage_coordinates = list(subimages.keys())
    
    origin_polygons = np.array(origin_polygons)
    origin_properties = np.array(origin_properties)

    transformed_polygons = origin_polygons.copy()
    for subimage_coordinate in subimage_coordinates:
        keep = bstool.select_polygons_in_range(origin_polygons, subimage_coordinate, image_size=(subsize, subsize))

        transformed_polygons[keep] = np.array(bstool.chang_polygon_coordinate(origin_polygons[keep].copy(), subimage_coordinate))

        transformed_polygons[keep] = np.array(bstool.clip_boundary_polygon(transformed_polygons[keep], image_size=(subsize, subsize)))

        drop = bstool.drop_subimage(subimages, subimage_coordinate, transformed_polygons[keep])

        if drop:
            continue

        subimage_properties = origin_properties[keep]
        subimage_polygons = transformed_polygons[keep]

        # bstool.show_polygons_on_image(subimages[subimage_coordinate], subimage_polygons)

        json_file = os.path.join(json_save_dir, '{}__{}_{}.json'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]))
        subimage_file = os.path.join(image_save_dir, '{}__{}_{}.png'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]))

        image_info = {"ori_filename": f"{file_name}.jpg",
                    "subimage_filename": '{}__{}_{}.png'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]),
                    "width": 1024,
                    "height": 1024,
                    "city": 'shanghai',
                    "sub_fold": 'arg',
                    "coordinate": [int(_) for _ in subimage_coordinate]}

        bstool.bs_json_dump(subimage_polygons.tolist(), subimage_properties.tolist(), image_info, json_file)
        cv2.imwrite(subimage_file, subimages[subimage_coordinate])