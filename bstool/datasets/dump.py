import json
from shapely import affinity
import tqdm
import pandas

import bstool

def bs_vis_json_dump(roof_polygons, footprint_polygons, offsets, json_file):
    annos = []
    for roof_polygon, footprint_polygon, offset in zip(roof_polygons, footprint_polygons, offsets):
        offset = [float(_) for _ in offset]
        object_struct = dict()
        if roof_polygon.geom_type == 'MultiPolygon':
            for roof_polygon_ in roof_polygon:
                if roof_polygon_.area < 20:
                    continue
                else:
                    object_struct['roof'] = bstool.polygon2mask(roof_polygon_)
                    object_struct['footprint'] = bstool.polygon2mask(footprint_polygon)
                    object_struct['offset'] = offset
                    annos.append(object_struct)
        elif roof_polygon.geom_type == 'Polygon':
            object_struct['roof'] = bstool.polygon2mask(roof_polygon)
            object_struct['footprint'] = bstool.polygon2mask(footprint_polygon)
            object_struct['offset'] = offset
        
            annos.append(object_struct)
        else:
            continue

    json_data = {"annotations": annos}

    with open(json_file, "w") as jsonfile:
        json.dump(json_data, jsonfile, indent=4)


def bs_json_dump(polygons, properties, image_info, json_file):
    """dump json file designed for building segmentation

    Args:
        polygons (list): list of polygons
        properties (list): list of property
        image_info (dict): image information
        json_file (str): json file name
    """
    annos = []
    for idx, (roof_polygon, single_property) in enumerate(zip(polygons, properties)):
        object_struct = dict()
        if roof_polygon.geom_type == 'MultiPolygon':
            for roof_polygon_ in roof_polygon:
                if roof_polygon_.area < 20:
                    continue
                else:
                    object_struct['roof'] = bstool.polygon2mask(roof_polygon_)
                    xoffset, yoffset = single_property['xoffset'], single_property['yoffset']
                    transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
                    footprint_polygon = affinity.affine_transform(roof_polygon_, transform_matrix)
                    if 'Floor' in single_property.keys():
                        if single_property['Floor'] is None:
                            building_height = 0.0
                        else:
                            building_height = 3 * single_property['Floor']
                    elif 'half_H' in single_property.keys():
                        if single_property['half_H'] is None:
                            building_height = 0.0
                        else:
                            building_height = single_property['half_H']
                    else:
                        raise(RuntimeError("No Floor key in property, keys = {}".format(single_property.keys())))    
                    object_struct['footprint'] = bstool.polygon2mask(footprint_polygon)
                    object_struct['offset'] = [xoffset, yoffset]
                    object_struct['ignore'] = single_property['ignore']
                    object_struct['building_height'] = building_height

                    annos.append(object_struct)
        elif roof_polygon.geom_type == 'Polygon':
            object_struct['roof'] = bstool.polygon2mask(roof_polygon)
            xoffset, yoffset = single_property['xoffset'], single_property['yoffset']
            transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
            footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)
            if 'Floor' in single_property.keys():
                if single_property['Floor'] is None:
                    building_height = 0.0
                else:
                    building_height = 3 * single_property['Floor']
            elif 'half_H' in single_property.keys():
                if single_property['half_H'] is None:
                    building_height = 0.0
                else:
                    building_height = single_property['half_H']
            else:
                raise(RuntimeError("No Floor key in property, keys = {}".format(single_property.keys())))
            object_struct['footprint'] = bstool.polygon2mask(footprint_polygon)
            object_struct['offset'] = [xoffset, yoffset]
            object_struct['ignore'] = single_property['ignore']
            object_struct['building_height'] = building_height
        
            annos.append(object_struct)
        else:
            continue
            # print("Runtime Warming: This processing do not support {}".format(type(roof_polygon)))

    json_data = {"image": image_info,
                "annotations": annos
                }

    with open(json_file, "w") as jsonfile:
        json.dump(json_data, jsonfile, indent=4)

def bs_json_dump_v2(polygons, properties, image_info, json_file):
    """dump json file designed for building segmentation (for lingxuan)

    Args:
        polygons (list): list of polygons (footprint polygons)
        properties (list): list of property
        image_info (dict): image information
        json_file (str): json file name
    """
    annos = []
    for idx, (footprint_polygon, single_property) in enumerate(zip(polygons, properties)):
        object_struct = dict()
        if footprint_polygon.geom_type == 'MultiPolygon':
            for footprint_polygon_ in footprint_polygon:
                if footprint_polygon_.area < 20:
                    continue
                else:
                    object_struct['footprint'] = bstool.polygon2mask(footprint_polygon_)
                    xoffset, yoffset = single_property['xoffset'], single_property['yoffset']
                    offset = [xoffset, yoffset]
                    roof_polygon = bstool.footprint2roof_single(footprint_polygon_, offset, offset_model='footprint2roof')
                    if 'Floor' in single_property.keys():
                        if single_property['Floor'] is None:
                            building_height = 0.0
                        else:
                            building_height = 3 * single_property['Floor']
                    elif 'half_H' in single_property.keys():
                        if single_property['half_H'] is None:
                            building_height = 0.0
                        else:
                            building_height = single_property['half_H']
                    else:
                        raise(RuntimeError("No Floor key in property, keys = {}".format(single_property.keys())))    
                    object_struct['roof'] = bstool.polygon2mask(roof_polygon)
                    object_struct['offset'] = [xoffset, yoffset]
                    object_struct['ignore'] = single_property['ignore']
                    object_struct['building_height'] = building_height

                    annos.append(object_struct)
        elif footprint_polygon.geom_type == 'Polygon':
            object_struct['footprint'] = bstool.polygon2mask(footprint_polygon)
            xoffset, yoffset = single_property['xoffset'], single_property['yoffset']
            offset = [xoffset, yoffset]
            if not (isinstance(xoffset, float) and isinstance(yoffset, float)):
                print("offset: ", offset)
                return None
                # continue
            roof_polygon = bstool.footprint2roof_single(footprint_polygon, offset, offset_model='footprint2roof')
            if 'Floor' in single_property.keys():
                if single_property['Floor'] is None:
                    building_height = 0.0
                else:
                    building_height = 3 * single_property['Floor']
            elif 'half_H' in single_property.keys():
                if single_property['half_H'] is None:
                    building_height = 0.0
                else:
                    building_height = single_property['half_H']
            else:
                raise(RuntimeError("No Floor key in property, keys = {}".format(single_property.keys())))
            object_struct['roof'] = bstool.polygon2mask(roof_polygon)
            object_struct['offset'] = [xoffset, yoffset]
            object_struct['ignore'] = single_property['ignore']
            object_struct['building_height'] = building_height
        
            annos.append(object_struct)
        else:
            continue
            # print("Runtime Warming: This processing do not support {}".format(type(roof_polygon)))

    json_data = {"image": image_info,
                "annotations": annos
                }

    with open(json_file, "w") as jsonfile:
        json.dump(json_data, jsonfile, indent=4)
    
    return True

def urban3d_json_dump(polygons, properties, image_info, json_file):
    """dump json file designed for building segmentation

    Args:
        polygons (list): list of polygons
        properties (list): list of property
        image_info (dict): image information
        json_file (str): json file name
    """
    annos = []
    for idx, (roof_polygon, single_property) in enumerate(zip(polygons, properties)):
        object_struct = dict()
        if roof_polygon.geom_type == 'MultiPolygon':
            for roof_polygon_ in roof_polygon:
                if roof_polygon_.area < 20:
                    continue
                else:
                    object_struct['roof'] = bstool.polygon2mask(roof_polygon_)
                    xoffset, yoffset = single_property['offset']
                    transform_matrix = [1, 0, 0, 1, xoffset, yoffset]
                    footprint_polygon = affinity.affine_transform(roof_polygon_, transform_matrix)
                    building_height = single_property['building_height']
                    object_struct['footprint'] = bstool.polygon2mask(footprint_polygon)
                    object_struct['offset'] = [xoffset, yoffset]
                    object_struct['building_height'] = building_height

                    annos.append(object_struct)
        elif roof_polygon.geom_type == 'Polygon':
            object_struct['roof'] = bstool.polygon2mask(roof_polygon)
            xoffset, yoffset = single_property['offset']
            transform_matrix = [1, 0, 0, 1, xoffset, yoffset]
            footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)
            building_height = single_property['building_height']
            object_struct['footprint'] = bstool.polygon2mask(footprint_polygon)
            object_struct['offset'] = [xoffset, yoffset]
            object_struct['building_height'] = building_height
        
            annos.append(object_struct)
        else:
            continue
            # print("Runtime Warming: This processing do not support {}".format(type(roof_polygon)))

    json_data = {"image": image_info,
                "annotations": annos
                }

    with open(json_file, "w") as jsonfile:
        json.dump(json_data, jsonfile, indent=4)

def bs_csv_dump(objects, roof_csv_file, footprint_csv_file):
    ori_image_name_list = list(objects.keys())

    first_in = True
    for ori_image_name in tqdm.tqdm(ori_image_name_list):
        buildings = objects[ori_image_name]

        if len(buildings) == 0:
            continue
        
        roof_polygons = [building['roof_polygon'] for building in buildings]
        footprint_polygons = [building['footprint_polygon'] for building in buildings]
        scores = [building['score'] for building in buildings]
        heights = [building['height'] for building in buildings]
        offsets = [building['offset'] for building in buildings]

        roof_csv = pandas.DataFrame({'ImageId': ori_image_name,
                                      'BuildingId': range(len(roof_polygons)),
                                      'PolygonWKT_Pix': roof_polygons,
                                      'Confidence': scores,
                                      'Offset': offsets,
                                      'Height': heights})
        footprint_csv = pandas.DataFrame({'ImageId': ori_image_name,
                                      'BuildingId': range(len(footprint_polygons)),
                                      'PolygonWKT_Pix': footprint_polygons,
                                      'Confidence': scores,
                                      'Offset': offsets,
                                      'Height': heights})
        if first_in:
            roof_csv_dataset = roof_csv
            footprint_csv_dataset = footprint_csv
            first_in = False
        else:
            roof_csv_dataset = roof_csv_dataset.append(roof_csv)
            footprint_csv_dataset = footprint_csv_dataset.append(footprint_csv)

    roof_csv_dataset.to_csv(roof_csv_file, index=False)
    footprint_csv_dataset.to_csv(footprint_csv_file, index=False)