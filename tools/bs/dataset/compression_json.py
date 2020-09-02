import bstool
import mmcv
import json

if __name__ == '__main__':
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    for city in cities:
        src_json_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_train_{city}.json'
        dst_json_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_train_{city}_only_footprint.json'

        src_data = mmcv.load(src_json_file)
        src_annotations = src_data['annotations']

        dst_annotations = []
        for src_annotation in src_annotations:
            src_annotation['only_footprint'] = 1

            dst_annotations.append(src_annotation)

        src_data['annotations'] = dst_annotations

        with open(dst_json_file, "w") as jsonfile:
            json.dump(src_data, jsonfile, sort_keys=True, indent=4)