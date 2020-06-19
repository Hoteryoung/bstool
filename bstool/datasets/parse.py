import numpy as np
import cv2
import geopandas
import rasterio as rio
import shapely
from shapely.geometry import Polygon, MultiPolygon

import bstool


class ShpParse():
    """Class for parse shapefile (just parse)
    """
    def __call__(self,
                shp_file,
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

class MaskParse():
    def __call__(self,
                 mask_file,
                 subclasses=(1, 3)):
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

            xmin, ymin, xmax, ymax = polygon.bounds

            object_struct['mask'] = bstool.polygon2mask(polygon)
            object_struct['polygon'] = polygon
            objects.append(object_struct)

        return objects
        