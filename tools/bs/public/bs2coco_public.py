import os
import cv2
import json
import argparse
import numpy as np

import bstool


class BS2COCO(bstool.Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """generation coco annotation for COCO dataset

        Args:
            annotpath (str): annotation file
            imgpath (str): image file

        Returns:
            dict: annotation information
        """
        objects = self.__json_parse__(annotpath, imgpath)
        
        coco_annotations = []

        for object_struct in objects:
            bbox = object_struct['bbox']
            # segmentation = object_struct['segmentation']
            segmentation = object_struct['footprint_mask']
            label = object_struct['label']

            roof_bbox = object_struct['roof_bbox']
            building_bbox = object_struct['building_bbox']
            roof_mask = object_struct['roof_mask']
            footprint_bbox = object_struct['footprint_bbox']
            footprint_mask = object_struct['footprint_mask']
            ignore_flag = object_struct['ignore_flag']
            offset = object_struct['offset']
            building_height = object_struct['building_height']
            iscrowd = object_struct['iscrowd']

            width = bbox[2]
            height = bbox[3]
            area = bstool.mask2polygon(segmentation).area

            if area <= self.small_object_area and self.groundtruth:
                self.small_object_idx += 1
                continue

            coco_annotation = {}
            coco_annotation['bbox'] = bbox
            coco_annotation['segmentation'] = [segmentation]
            coco_annotation['category_id'] = label
            coco_annotation['area'] = np.float(area)

            coco_annotation['roof_bbox'] = roof_bbox
            coco_annotation['building_bbox'] = building_bbox
            coco_annotation['roof_mask'] = roof_mask
            coco_annotation['footprint_bbox'] = footprint_bbox
            coco_annotation['footprint_mask'] = footprint_mask
            coco_annotation['ignore_flag'] = ignore_flag
            coco_annotation['offset'] = offset
            coco_annotation['building_height'] = building_height
            coco_annotation['iscrowd'] = iscrowd

            coco_annotations.append(coco_annotation)

        return coco_annotations
    
    def __json_parse__(self, label_file, image_file):
        objects = []
        if self.groundtruth:
            objects = bstool.bs_json_parse(label_file, fix_height=fix_height)
            
            if with_height_sample:
                heights = [obj['building_height'] for obj in objects]
                if max(heights) < min_height:
                    return []
                else:
                    print("The max height is: ", max(heights))
        else:
            object_struct = {}
            object_struct['bbox'] = [0, 0, 0, 0]
            object_struct['roof_bbox'] = [0, 0, 0, 0]
            object_struct['footprint_bbox'] = [0, 0, 0, 0]
            object_struct['building_bbox'] = [0, 0, 0, 0]

            object_struct['roof_mask'] = [0, 0, 0, 0, 0, 0, 0, 0]
            object_struct['footprint_mask'] = [0, 0, 0, 0, 0, 0, 0, 0]
            object_struct['ignore_flag'] = 0
            object_struct['offset'] = [0, 0]
            object_struct['building_height'] = 0
            
            object_struct['segmentation'] = [0, 0, 0, 0, 0, 0, 0, 0]
            object_struct['label'] = 1
            object_struct['iscrowd'] = 0
            objects.append(object_struct)

        return objects

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval on semantic segmentation')
    parser.add_argument(
        '--city',
        type=str,
        default='None', 
        help='dataset for evaluation')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.city == 'None':
        cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    else:
        cities = [args.city]
    # basic dataset information
    info = {"year" : 2020,
            "version" : "1.0",
            "description" : "Building-Segmentation-COCO",
            "contributor" : "Jinwang Wang",
            "url" : "jwwangchn@gmail.com",
            "date_created" : "2020"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    original_class = {'building': 1}

    converted_class = [{'supercategory': 'none', 'id': 1,  'name': 'building',                   }]

    # dataset's information
    image_format='.png'
    anno_format='.json'

    # dataset meta data
    core_dataset_name = 'buildchange'
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    # cities = ['jinan', 'haerbin', 'chengdu']
    # cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu', 'xian_fine', 'dalian_fine']
    cities = ['shanghai_fine']
    # cities = ['shanghai_xian']
    version = '20201028'
    release_version = f'public/{version}'

    groundtruth = True
    with_height_sample = False
    min_height = 100

    for idx, city in enumerate(cities):
        print(f"Begin to process {city} data!")
        if 'xian' in city or 'shanghai_fine' in city:
            anno_name = [core_dataset_name, f'public_{version}', 'val', city, 'minarea_500']
            fix_height = True
        else:
            anno_name = [core_dataset_name, f'public_{version}', 'train', city, 'minarea_500']
            fix_height = True

        if with_height_sample:
            anno_name.append("height_sampled")
        
        imgpath = f'./data/{core_dataset_name}/{release_version}/{city}/images'
        annopath = f'./data/{core_dataset_name}/{release_version}/{city}/labels'
        save_path = f'./data/{core_dataset_name}/{release_version}/coco/annotations'
        
        bstool.mkdir_or_exist(save_path)

        bs2coco = BS2COCO(imgpath=imgpath,
                                annopath=annopath,
                                image_format=image_format,
                                anno_format=anno_format,
                                data_categories=converted_class,
                                data_info=info,
                                data_licenses=licenses,
                                data_type="instances",
                                groundtruth=groundtruth,
                                small_object_area=500,  # 10 for old
                                image_size=(1024, 1024))

        images, annotations = bs2coco.get_image_annotation_pairs()

        json_data = {"info" : bs2coco.info,
                    "images" : images,
                    "licenses" : bs2coco.licenses,
                    "type" : bs2coco.type,
                    "annotations" : annotations,
                    "categories" : bs2coco.categories}

        with open(os.path.join(save_path, "_".join(anno_name) + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)