import json
from shapely import affinity

import bstool


def bs_json_dump(polygons, properties, image_info, json_file):
    annos = []
    for idx, (roof_polygon, single_property) in enumerate(zip(polygons, properties)):
        object_struct = dict()
        object_struct['roof'] = bstool.polygon2mask(roof_polygon)
        
        xoffset, yoffset = single_property['xoffset'], single_property['yoffset']
        transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
        footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)    
        object_struct['footprint'] = bstool.polygon2mask(footprint_polygon)
        object_struct['offset'] = [xoffset, yoffset]
        object_struct['ignore'] = single_property['ignore']

        annos.append(object_struct)

    json_data = {"image": image_info,
                "annotations": annos
                }

    with open(json_file, "w") as jsonfile:
        json.dump(json_data, jsonfile, indent=4)