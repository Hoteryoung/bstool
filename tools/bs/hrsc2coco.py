# -*- encoding: utf-8 -*-
'''
@File    :   hrsc2coco.py
@Time    :   2020/12/30 22:38:48
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   将 HRSC 转换为 COCO 格式
'''

import argparse

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET

import bstool


class HRSC2COCO(bstool.Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__dataset_parse__(annotpath, imgpath)
        
        coco_annotations = []
        
        for object_struct in objects:
            bbox = object_struct['bbox']
            label = object_struct['label']
            segmentation = object_struct['segmentation']
            pointobb = object_struct['pointobb']
            thetaobb = object_struct['thetaobb']
            hobb = object_struct['hobb']

            width = bbox[2]
            height = bbox[3]
            area = height * width

            if area <= self.small_object_area and self.groundtruth:
                self.small_object_idx += 1
                continue

            coco_annotation = {}
            coco_annotation['bbox'] = bbox
            coco_annotation['category_id'] = label
            coco_annotation['area'] = np.float(area)
            coco_annotation['segmentation'] = [segmentation]
            coco_annotation['pointobb'] = pointobb
            coco_annotation['thetaobb'] = thetaobb
            coco_annotation['hobb'] = hobb

            coco_annotations.append(coco_annotation)
            
        return coco_annotations
    
    def __dataset_parse__(self, label_file, image_file):
        objects = []
        if self.groundtruth:
            tree = ET.parse(label_file)
            root = tree.getroot()
            objects = []
            hrsc_object = root.find('HRSC_Objects')
            for hrsc_sub_object in hrsc_object.findall('HRSC_Object'):
                obj_struct = {}
                xmin = float(hrsc_sub_object.find('box_xmin').text)
                ymin = float(hrsc_sub_object.find('box_ymin').text)
                xmax = float(hrsc_sub_object.find('box_xmax').text)
                ymax = float(hrsc_sub_object.find('box_ymax').text)
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin

                cx = float(hrsc_sub_object.find('mbox_cx').text)
                cy = float(hrsc_sub_object.find('mbox_cy').text)
                rbbox_w = float(hrsc_sub_object.find('mbox_w').text)
                rbbox_h = float(hrsc_sub_object.find('mbox_h').text)
                angle = float(hrsc_sub_object.find('mbox_ang').text)

                obj_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
                obj_struct['thetaobb'] = [cx, cy, rbbox_w, rbbox_h, angle]
                obj_struct['segmentation'] = bstool.thetaobb2pointobb(obj_struct['thetaobb'])
                obj_struct['pointobb'] = bstool.pointobb_best_point_sort(obj_struct['segmentation'])
                obj_struct['keypoints'] = obj_struct['pointobb'][:]
                for idx in [2, 5, 8, 11]:
                    obj_struct['keypoints'].insert(idx, 2)
                obj_struct['hobb'] = bstool.thetaobb2hobb(obj_struct['thetaobb'], bstool.pointobb_best_point_sort)
                obj_struct['label'] = 1

                objects.append(obj_struct)
        else:
            obj_struct = {}
            obj_struct['segmentation'] = [0, 0, 0, 0, 0, 0, 0, 0]
            obj_struct['keypoint'] = [0, 0, 0, 0, 0, 0, 0, 0]
            obj_struct['pointobb'] = [0, 0, 0, 0, 0, 0, 0, 0]
            obj_struct['thetaobb'] = [0, 0, 0, 0, 0]
            obj_struct['hobb'] = [0, 0, 0, 0, 0]
            obj_struct['bbox'] = [0, 0, 0, 0]
            obj_struct['label'] = 0

            objects.append(obj_struct)
        return objects

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument(
        '--imagesets',
        type=str,
        nargs='+',
        choices=['trainval', 'test'])
    parser.add_argument(
        '--release_version', default='v1', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # basic dataset information
    info = {"year" : 2019,
                "version" : "1.0",
                "description" : "HRSC2016-COCO",
                "contributor" : "Jinwang Wang",
                "url" : "jwwangchn.cn",
                "date_created" : "2019"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    image_format='.bmp'
    anno_format='.xml'

    original_class = {'ship': 1}

    converted_class = [{'supercategory': 'none', 'id': 1,  'name': 'ship',                },]

    meta_info = dict(core_dataset_name = 'hrsc',
                     imagesets = ['trainval', 'test'],  # 'trainval', 'test'
                     release_version = 'v1',
                     with_groundtruth=True)

    meta_info = bstool.dotdict(meta_info)
    
    anno_name = [meta_info.core_dataset_name, meta_info.release_version]

    for idx, imageset in enumerate(meta_info.imagesets):
        if idx == 0:
            anno_name.append(imageset)
        else:
            anno_name[-1] = imageset

        imgpath = './data/{}/{}/{}/images'.format(meta_info.core_dataset_name, meta_info.release_version, imageset)
        annopath = './data/{}/{}/{}/annotations'.format(meta_info.core_dataset_name, meta_info.release_version, imageset)
        save_path = './data/{}/{}/coco/annotations'.format(meta_info.core_dataset_name, meta_info.release_version)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        imageset_file = './data/{}/{}/{}/{}.txt'.format(meta_info.core_dataset_name, meta_info.release_version, imageset, imageset)

        converter = HRSC2COCO(imgpath=imgpath,
                                annopath=annopath,
                                imageset_file=None,
                                image_format=image_format,
                                anno_format=anno_format,
                                data_categories=converted_class,
                                data_info=info,
                                data_licenses=licenses,
                                data_type="instances",
                                groundtruth=meta_info.with_groundtruth,
                                small_object_area=0,
                                meta_info=meta_info)

        images, annotations = converter.get_image_annotation_pairs()

        json_data = {"info" : converter.info,
                    "images" : images,
                    "licenses" : converter.licenses,
                    "type" : converter.type,
                    "annotations" : annotations,
                    "categories" : converter.categories}

        with open(os.path.join(save_path, "_".join(anno_name) + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
