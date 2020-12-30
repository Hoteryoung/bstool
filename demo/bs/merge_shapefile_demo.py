# -*- encoding: utf-8 -*-
'''
@File    :   merge_shapefile_demo.py
@Time    :   2020/12/30 21:58:54
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   由于标注的时候是分块标注的，导致边缘部分的建筑物被分开，因此使用此程序合并分块标注的标注信息
'''


import pandas
import geopandas

import bstool


if __name__ == '__main__':
    shp_file = '/data/buildchange/v0/shanghai/shp_4326/L18_106968_219320.shp'
    geo_file = '/data/buildchange/v0/shanghai/geo_info/L18_106968_219320.png'
    rgb_file = '/data/buildchange/v0/shanghai/images/L18_106968_219320.jpg'

    merged_shp_file = '/data/buildchange/demo/L18_106968_219320.shp'

    objects = bstool.shp_parse(shp_file=shp_file,
                                geo_file=geo_file,
                                src_coord='4326',
                                dst_coord='4326')

    polygons = [obj['polygon'] for obj in objects]
    properties = [obj['property'] for obj in objects]

    merged_polygons, merged_properties = bstool.merge_polygons(polygons, properties)

    converted_polygons = [bstool.polygon_coordinate_convert(merged_polygon, geo_file) for merged_polygon in merged_polygons]

    bstool.show_polygon(converted_polygons)

    properties = []

    for idx, merged_property in enumerate(merged_properties):
        merged_property['Id'] = idx
        properties.append(merged_property)
        
    df = pandas.DataFrame(properties)
    gdf = geopandas.GeoDataFrame(df, geometry=merged_polygons, crs='EPSG:4326')
    gdf.to_file(merged_shp_file, encoding='utf-8')