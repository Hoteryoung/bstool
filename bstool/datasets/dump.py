import json
from shapely import affinity

import bstool


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
                        raise(RuntimeError("No Floor key in property, keys = {}".format(type(single_property.keys()))))    
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
                raise(RuntimeError("No Floor key in property, keys = {}".format(type(single_property.keys()))))    
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