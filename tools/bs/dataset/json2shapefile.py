import numpy as np
import os
import pandas
import geopandas
import rasterio as rio
import mmcv
import gdal
import bstool


def json_parser(json_file):
    """parse the json file generated by lingxuan

    Args:
        json_file (str): file path
    """
    content = mmcv.load(json_file)
    objects = []
    building_num = len(content['foot'])
    for idx in range(building_num):
        object_struct = dict()
        footprint_mask = content['foot'][str(idx)]
        footprint_mask = np.array(footprint_mask).reshape(1, -1).tolist()[0]
        footprint_polygon = bstool.mask2polygon(footprint_mask)

        building_height = content['buildHeight'][str(idx)]
        
        xoffset = content['xyoffset'][str(idx)][0]
        yoffset = content['xyoffset'][str(idx)][1]

        object_struct['polygon'] = footprint_polygon
        object_struct['property'] = {"Id": idx, "Floor": building_height // 3, "xoffset": xoffset, "yoffset": yoffset}

        objects.append(object_struct)
    
    return objects

def coord_transform(geo_file, polygons):
    dataset = gdal.Open(geo_file)
    transform = dataset.GetGeoTransform()

    converted_polygons = []
    for polygon in polygons:
        converted_polygon = []
        for c in polygon.exterior.coords:
            converted_polygon.append(transform[0] + transform[1] * c[0] + c[1] * transform[2])
            converted_polygon.append(transform[3] + transform[4] * c[0] + c[1] * transform[5])
        converted_polygon = bstool.mask2polygon(converted_polygon)

        converted_polygons.append(converted_polygon)
    
    return converted_polygons

def shapefile_dump(objects, geo_file, shp_file):
    polygons = [obj['polygon'] for obj in objects]
    properties = [obj['property'] for obj in objects]
    
    # converted_polygons = [bstool.polygon_coordinate_convert(polygon, geo_file, src_coord='pixel', dst_coord='4326') for polygon in polygons]
    converted_polygons = coord_transform(geo_file, polygons)

    df = pandas.DataFrame(properties)
    gdf = geopandas.GeoDataFrame(df, geometry=converted_polygons, crs='EPSG:4326')
    gdf.to_file(shp_file, encoding='utf-8')


if __name__ == "__main__":
    cities = ['shanghai', 'beijing', 'jinan', 'chengdu', 'haerbin']
    sub_folds = {'beijing':  ['arg', 'google', 'ms', 'tdt'],
                 'chengdu':  ['arg', 'google', 'ms', 'tdt'],
                 'haerbin':  ['arg', 'google', 'ms'],
                 'jinan':    ['arg', 'google', 'ms', 'tdt'],
                 'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}
    
    for city in cities:
        for sub_fold in sub_folds[city]:
            json_dir = f'/mnt/lustre/menglingxuan/buildingwolf/20200329/{city}_18/{sub_fold}/anno_20200924/OffsetField/TXT'
            geo_dir = f'/mnt/lustre/wangjinwang/buildchange/codes/bstool/data/buildchange/v0/{city}/{sub_fold}/geo_info'
            shp_dir = f'/mnt/lustre/wangjinwang/buildchange/codes/bstool/data/buildchange/v0/{city}/{sub_fold}/footprint_shp_4326_20200924'

            bstool.mkdir_or_exist(shp_dir)

            for json_file_name in os.listdir(json_dir):
                print(f'Processing {city} {sub_fold} {json_file_name}')
                basename = bstool.get_basename(json_file_name)

                json_file = os.path.join(json_dir, basename + '.json')
                geo_file = os.path.join(geo_dir, basename + '.png')
                shp_file = os.path.join(shp_dir, basename + '.shp')

                objects = json_parser(json_file)
                shapefile_dump(objects, geo_file, shp_file)
    