import os
import cv2
import json
import numpy as np

import bstool


class Urban3D2COCO(bstool.Convert2COCO):
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
            segmentation = object_struct['segmentation']
            label = object_struct['label']

            roof_bbox = object_struct['roof_bbox']
            building_bbox = object_struct['building_bbox']
            roof_mask = object_struct['roof_mask']
            footprint_bbox = object_struct['footprint_bbox']
            footprint_mask = object_struct['footprint_mask']
            offset = object_struct['offset']
            building_height = object_struct['building_height']

            width = bbox[2]
            height = bbox[3]
            area = height * width

            footprint_bbox_width, footprint_bbox_height = footprint_bbox[2], footprint_bbox[3]
            if footprint_bbox_width * footprint_bbox_height <= self.small_object_area and self.groundtruth:
                self.small_object_idx += 1
                continue

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
            coco_annotation['offset'] = offset
            coco_annotation['building_height'] = building_height

            coco_annotations.append(coco_annotation)

        return coco_annotations
    
    def __json_parse__(self, label_file, image_file):
        objects = []
        if self.groundtruth:
            objects = bstool.urban3d_json_parse(label_file)
            
        else:
            object_struct = {}
            object_struct['bbox'] = [0, 0, 0, 0]
            object_struct['roof_bbox'] = [0, 0, 0, 0]
            object_struct['footprint_bbox'] = [0, 0, 0, 0]
            object_struct['building_bbox'] = [0, 0, 0, 0]

            object_struct['roof_mask'] = [0, 0, 0, 0, 0, 0, 0, 0]
            object_struct['footprint_mask'] = [0, 0, 0, 0, 0, 0, 0, 0]
            object_struct['offset'] = [0, 0]
            object_struct['building_height'] = 0
            
            object_struct['segmentation'] = [0, 0, 0, 0, 0, 0, 0, 0]
            object_struct['label'] = 1
            objects.append(object_struct)

        return objects


if __name__ == "__main__":
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
    core_dataset_name = 'urban3d'
    imagesets = ['val', 'train']
    # imagesets = ['val']
    # datasets = ['JAX_OMA', 'ATL']
    datasets = ['JAX_OMA']
    release_version = 'v2'

    groundtruth = True
    min_height = 100
    image_size = 1024

    for imageset in imagesets:
        for dataset in datasets:
            print(f"Begin to process {imageset} data!")
            
            anno_name = [core_dataset_name, release_version, imageset, dataset]
            
            imgpath = f'./data/{core_dataset_name}/{release_version}/{imageset}/images'
            annopath = f'./data/{core_dataset_name}/{release_version}/{imageset}/labels'
            save_path = f'./data/{core_dataset_name}/{release_version}/coco/annotations'

            imageset_file = f'./data/{core_dataset_name}/{release_version}/{imageset}/{dataset}_imageset_file.txt'
            
            bstool.mkdir_or_exist(save_path)

            dataset2coco = Urban3D2COCO(imgpath=imgpath,
                                        annopath=annopath,
                                        imageset_file=imageset_file,
                                        image_format=image_format,
                                        anno_format=anno_format,
                                        data_categories=converted_class,
                                        data_info=info,
                                        data_licenses=licenses,
                                        data_type="instances",
                                        groundtruth=groundtruth,
                                        small_object_area=10,
                                        image_size=(image_size, image_size))

            images, annotations = dataset2coco.get_image_annotation_pairs()

            json_data = {"info" : dataset2coco.info,
                        "images" : images,
                        "licenses" : dataset2coco.licenses,
                        "type" : dataset2coco.type,
                        "annotations" : annotations,
                        "categories" : dataset2coco.categories}

            with open(os.path.join(save_path, "_".join(anno_name) + ".json"), "w") as jsonfile:
                json.dump(json_data, jsonfile, sort_keys=True, indent=4)