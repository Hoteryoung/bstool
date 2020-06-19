import numpy as np
import cv2
import geopandas
import rasterio as rio
import shapely
from shapely.geometry import Polygon, MultiPolygon

import mmcv
import bstool


def shp_parse(shp_file,
              geo_file,
              src_coord='4326',
              dst_coord='pixel'):
    """parse shapefile

    Args:
        shp_file (str): shapefile
        geo_file (str or rio class): geometry information
        src_coord (str, optional): source coordinate system. Defaults to '4326'.
        dst_coord (str, optional): destination coordinate system. Defaults to 'pixel'.

    Returns:
        list: parsed objects
    """
    try:
        shp = geopandas.read_file(shp_file, encoding='utf-8')
    except:
        print("Can't open this shapefile: {}".format(shp_file))
        return []

    if isinstance(geo_file, str):
        geo_img = rio.open(geo_file)
    
    dst_polygons = []
    dst_properties = []
    for idx, row_data in shp.iterrows():
        src_polygon = row_data.geometry
        src_property = row_data[:-1]

        if src_polygon.geom_type == 'Polygon':
            dst_polygons.append(bstool.polygon_coordinate_convert(src_polygon, geo_img, src_coord, dst_coord))
            dst_properties.append(src_property.to_dict())
        elif src_polygon.geom_type == 'MultiPolygon':
            for sub_polygon in src_polygon:
                dst_polygons.append(bstool.polygon_coordinate_convert(sub_polygon, geo_img, src_coord, dst_coord))
                dst_properties.append(src_property.to_dict())
        else:
            raise(RuntimeError("type(src_polygon) = {}".format(type(src_polygon))))
    
    objects = []
    for idx, (dst_polygon, dst_property) in enumerate(zip(dst_polygons, dst_properties)):
        object_struct = dict()

        object_struct['mask'] = bstool.polygon2mask(dst_polygon)
        xmin, ymin, xmax, ymax = dst_polygon.bounds
        object_struct['bbox'] = [xmin, ymin, xmax, ymax]
        object_struct['polygon'] = dst_polygon
        object_struct['property'] = dst_property

        objects.append(object_struct)

    return objects

def mask_parse(mask_file,
               subclasses=(1, 3)):
    """parse mask image

    Args:
        mask_file (str or np.array): mask image
        subclasses (tuple, optional): parse which class. Defaults to (1, 3).

    Returns:
        list: list of objects
    """
    if isinstance(mask_file, str):
        mask_image = cv2.imread(mask_file)

    if mask_image is None:
        if isinstance(mask_file, str):
            print("Can not open this mask (ignore) file: {}".format(mask_file))
        else:
            print("Can not handle mask image (np.array), it is empty")
        return []

    sub_mask = bstool.generate_subclass_mask(mask_image, subclasses=subclasses)
    polygons = bstool.generate_polygon(sub_mask)

    objects = []
    for polygon in polygons:
        object_struct = dict()
        object_struct['mask'] = bstool.polygon2mask(polygon)
        object_struct['polygon'] = polygon
        objects.append(object_struct)

    return objects
        
def bs_json_parse(json_file):
    annotations = mmcv.load(json_file)['annotations']
    objects = []
    for annotation in annotations:
        object_struct = {}
        roof_mask = annotation['roof']
        roof_polygon = bstool.mask2polygon(roof_mask)
        roof_bound = roof_polygon.bounds
        footprint_mask = annotation['footprint']
        footprint_polygon = bstool.mask2polygon(footprint_mask)
        footprint_bound = footprint_polygon.bounds
        building_xmin = np.minimum(roof_bound[0], footprint_bound[0])
        building_ymin = np.minimum(roof_bound[1], footprint_bound[1])
        building_xmax = np.maximum(roof_bound[2], footprint_bound[2])
        building_ymax = np.maximum(roof_bound[3], footprint_bound[3])

        building_bound = [building_xmin, building_ymin, building_xmax, building_ymax]

        xmin, ymin, xmax, ymax = list(roof_bound)
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
        object_struct['roof_bbox'] = object_struct['bbox']
        xmin, ymin, xmax, ymax = list(building_bound)
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        object_struct['building_bbox'] = [xmin, ymin, bbox_w, bbox_h]

        object_struct['roof_mask'] = roof_mask
        object_struct['footprint_mask'] = footprint_mask
        object_struct['ignore_flag'] = annotation['ignore']
        object_struct['offset'] = annotation['offset']
        
        object_struct['segmentation'] = roof_mask
        object_struct['label'] = 1
        object_struct['iscrowd'] = object_struct['ignore_flag']
        
        objects.append(object_struct)
    
    return objects