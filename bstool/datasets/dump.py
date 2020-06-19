import json

import bstool


def bs_json_dump(polygons, properties, image_info, json_file):
    annos = []
    for idx, (single_polygon, single_property) in enumerate(zip(polygons, properties)):
        object_struct = dict()
        object_struct['roof'] = bstool.polygon2mask(single_polygon)
        object_struct['footprint'] = bstool.polygon2mask(single_polygon)
        object_struct['offset'] = [0, 0]
        object_struct['ignore'] = single_property['ignore']

        annos.append(object_struct)

    json_data = {"image": image_info,
                "annotations": annos
                }

    with open(json_file, "w") as jsonfile:
        json.dump(json_data, jsonfile, indent=4)