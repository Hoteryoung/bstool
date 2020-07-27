import os
import cv2
import json
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
            segmentation = object_struct['segmentation']
            label = object_struct['label']
            thetaobb = object_struct['thetaobb']
            iscrowd = object_struct['iscrowd']

            width = bbox[2]
            height = bbox[3]
            area = height * width

            bbox_width, bbox_height = bbox[2], bbox[3]
            if bbox_width * bbox_height <= self.small_object_area and self.groundtruth:
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

            coco_annotation['iscrowd'] = iscrowd

            coco_annotations.append(coco_annotation)

        return coco_annotations

    def __bs_json2oriented_lines(self, objects):
        converted_objects = []
        for object_struct in objects:
            segmentation = object_struct['segmentation']
            
            label = object_struct['label']
            iscrowd = object_struct['iscrowd']

            splited_objects = []
            lines = bstool.mask2lines(segmentation)
            for line in lines:
                splited_object_struct = {}
                if with_normal:
                    thetaobb = bstool.line2thetaobb(line, angle_mode='normal')
                    pointobb = bstool.thetaobb2pointobb(thetaobb)
                else:
                    thetaobb = bstool.line2thetaobb(line, angle_mode='atan')
                    pointobb = bstool.thetaobb2pointobb(thetaobb)
                # pointobb = bstool.line2pointobb(line)
                # thetaobb = bstool.pointobb2thetaobb(pointobb)

                bbox = bstool.pointobb2bbox(pointobb)

                splited_object_struct['bbox'] = bstool.xyxy2xywh(bbox)
                splited_object_struct['segmentation'] = pointobb
                splited_object_struct['thetaobb'] = thetaobb
                splited_object_struct['label'] = label
                splited_object_struct['iscrowd'] = iscrowd

                splited_objects.append(splited_object_struct)

            converted_objects += splited_objects
                
        return converted_objects
    
    def __json_parse__(self, label_file, image_file):
        objects = []
        if self.groundtruth:
            objects = bstool.bs_json_parse(label_file)
            objects = self.__bs_json2oriented_lines(objects)
            
            if with_height_sample:
                heights = [obj['building_height'] for obj in objects]
                if max(heights) < min_height:
                    return []
                else:
                    print("The max height is: ", max(heights))
        else:
            object_struct = {}
            object_struct['bbox'] = [0, 0, 0, 0]
            object_struct['segmentation'] = [0, 0, 0, 0, 0, 0, 0, 0]
            object_struct['thetaobb'] = [0, 0, 0, 0, 0]
            object_struct['label'] = 1
            object_struct['iscrowd'] = 0
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
    core_dataset_name = 'buildchange'
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu', 'xian_fine', 'dalian_fine']
    # cities = ['xian_fine_origin']
    release_version = 'v1'

    with_normal = True

    groundtruth = True
    with_height_sample = False
    min_height = 100

    for idx, city in enumerate(cities):
        print(f"Begin to process {city} data!")
        if 'xian' in city or 'dalian' in city:
            anno_name = [core_dataset_name, release_version, 'val', city, 'oriented_line']
        else:
            anno_name = [core_dataset_name, release_version, 'train', city, 'oriented_line']

        if with_normal:
            anno_name.append('normal')

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
                                small_object_area=10,
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