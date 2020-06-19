import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import geojson
import networkx as nx
import rasterio as rio
import geopandas


def polygon2mask(polygon):
    """convet polygon to mask

    Arguments:
        polygon {Polygon} -- input polygon (single polygon)

    Returns:
        list -- converted mask ([x1, y1, x2, y2, ...])
    """
    mask = np.array(polygon.exterior.coords, dtype=np.int64)[:-1].ravel().tolist()
    return mask

def polygon_coordinate_convert(polygon, 
                               geo_img, 
                               src_coord='4326', 
                               dst_coord='pixel'):
        """convert polygon of source coordinate to destination coordinate

        Args:
            polygon (Polygon): source polygon
            geo_img (str or rio class): coordinate infomation
            src_coord (str, optional): source coordinate. Defaults to '4326'.
            dst_coord (str, optional): destination coordinate. Defaults to 'pixel'.

        Returns:
            Polygon: converted polygon
        """
        if isinstance(geo_img, str):
            geo_img = rio.open(geo_img)

        if src_coord == '4326':
            if dst_coord == '4326':
                converted_polygon = polygon
            elif dst_coord == 'pixel':
                converted_polygon = [(geo_img.index(c[0], c[1])[1], geo_img.index(c[0], c[1])[0]) for c in polygon.exterior.coords]
                converted_polygon = Polygon(converted_polygon)
            else:
                raise(RuntimeError("Not support dst_coord = {}".format(dst_coord)))
        elif src_coord == 'pixel':
            converted_polygon = polygon
        else:
            raise(RuntimeError("Not support src_coord = {}".format(src_coord)))

        return converted_polygon

def merge_polygons(polygons, 
                   properties=None, 
                   connection_mode='floor',
                   merge_boundary=False):

        floors = []
        for single_property in properties:
            if 'Floor' in single_property.keys():
                floors.append(single_property['Floor'])
            elif 'half_H' in single_property.keys():
                floors.append(single_property['half_H'])
            else:
                raise(RuntimeError("No Floor key in property, keys = {}".format(type(single_property.keys()))))

        merged_polygons = []
        merged_properties = []

        num_polygon = len(polygons)
        node_list = range(num_polygon)
        link_list = []
        for i in range(num_polygon):
            for j in range(i + 1, num_polygon):
                polygon1, polygon2 = polygons[i], polygons[j]
                floor1, floor2 = floors[i], floors[j]
                if polygon1.is_valid and polygon2.is_valid:
                    inter = polygon1.intersection(polygon2)

                    if connection_mode == 'floor':
                        if (inter.area > 0.0 or inter.geom_type in ["LineString", "MULTILINESTRING"]) and floor1 == floor2:
                            link_list.append([i, j])
                    elif connection_mode == 'line':
                        if inter.geom_type in ["LineString", "MULTILINESTRING"]:
                            link_list.append([i, j])
                    else:
                        raise(RuntimeError("Wrong connection_mode flag: {}".format(connection_mode)))

                    if merge_boundary:
                        if polygon1.area > polygon2.area:
                            difference = polygon1.difference(polygon2)
                            polygons[i] = difference
                        else:
                            difference = polygon2.difference(polygon1)
                            polygons[j] = difference
                else:
                    continue
        
        G = nx.Graph()
        for node in node_list:
            G.add_node(node, properties=properties[node])

        for link in link_list:
            G.add_edge(link[0], link[1])

        for c in nx.connected_components(G):
            nodeSet = G.subgraph(c).nodes()
            nodeSet = list(nodeSet)

            if len(nodeSet) == 1:
                single_polygon = polygons[nodeSet[0]]
                merged_polygons.append(single_polygon)
                merged_properties.append(properties[nodeSet[0]])
            else:
                pre_merge = [polygons[node] for node in nodeSet]
                post_merge = unary_union(pre_merge)
                if post_merge.geom_type == 'MultiPolygon':
                    for sub_polygon in post_merge:
                        merged_polygons.append(sub_polygon)
                        merged_properties.append(properties[nodeSet[0]])
                else:
                    merged_polygons.append(post_merge)
                    merged_properties.append(properties[nodeSet[0]])
        
        return merged_polygons, merged_properties


def get_ignored_polygon_idx(origin_polygons, ignore_polygons):
    origin_polygons = geopandas.GeoSeries(origin_polygons)
    ignore_polygons = geopandas.GeoSeries(ignore_polygons)

    origin_df = geopandas.GeoDataFrame({'geometry': origin_polygons, 'foot_df':range(len(origin_polygons))})
    ignore_df = geopandas.GeoDataFrame({'geometry': ignore_polygons, 'ignore_df':range(len(ignore_polygons))})

    res_intersection = geopandas.overlay(origin_df, ignore_df, how='intersection')
    inter_dict = res_intersection.to_dict()
    ignore_indexes = list(set(inter_dict['foot_df'].values()))
    ignore_indexes.sort()

    return ignore_indexes

def add_ignore_flag_in_property(properties, ignore_indexes):
    ret_properties = []
    for idx, single_property in enumerate(properties):
        if idx in ignore_indexes:
            single_property['ignore'] = 1
        else:
            single_property['ignore'] = 0
        ret_properties.append(single_property)
    return ret_properties