import numpy as np
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