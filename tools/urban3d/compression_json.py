import bstool
import mmcv
import json


if __name__ == '__main__':
    src_json_file = './data/urban3d/v2/coco/annotations/urban3d_v2_val_JAX_OMA.json'
    # src_json_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_train_samples.json'
    dst_json_file = './data/urban3d/v2/coco/annotations/urban3d_v2_val_JAX_OMA_compression.json'
    # dst_json_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_train_samples_compression.json'

    src_data = mmcv.load(src_json_file)
    src_annotations = src_data['annotations']

    dst_annotations = []
    for src_annotation in src_annotations:
        src_annotation.pop('footprint_mask')
        src_annotation.pop('roof_mask')

        dst_annotations.append(src_annotation)

    src_data['annotations'] = dst_annotations

    with open(dst_json_file, "w") as jsonfile:
        json.dump(src_data, jsonfile, sort_keys=True, indent=4)