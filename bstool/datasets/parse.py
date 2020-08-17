import numpy as np
import cv2
import geopandas
import pandas
import rasterio as rio
import shapely
from shapely.geometry import Polygon, MultiPolygon
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from shapely import affinity
from collections import defaultdict
import tqdm
import ast

import mmcv
import bstool


def shp_parse(shp_file,
              geo_file,
              src_coord='4326',
              dst_coord='pixel',
              keep_polarity=True,
              clean_polygon_flag=False):
    """parse shapefile

    Args:
        shp_file (str): shapefile
        geo_file (str or rio class): geometry information
        src_coord (str, optional): source coordinate system. Defaults to '4326'.
        dst_coord (str, optional): destination coordinate system. Defaults to 'pixel'.

    Returns:
        list: parsed objects
    """
    try:
        shp = geopandas.read_file(shp_file, encoding='utf-8')
    except:
        print("Can't open this shapefile: {}".format(shp_file))
        return []

    if isinstance(geo_file, str):
        geo_img = rio.open(geo_file)
    
    dst_polygons = []
    dst_properties = []
    for idx, row_data in shp.iterrows():
        src_polygon = row_data.geometry
        src_property = row_data[:-1]

        if src_polygon.geom_type == 'Polygon':
            dst_polygons.append(bstool.polygon_coordinate_convert(src_polygon, geo_img, src_coord, dst_coord, keep_polarity))
            dst_properties.append(src_property.to_dict())
        elif src_polygon.geom_type == 'MultiPolygon':
            for sub_polygon in src_polygon:
                dst_polygons.append(bstool.polygon_coordinate_convert(sub_polygon, geo_img, src_coord, dst_coord, keep_polarity))
                dst_properties.append(src_property.to_dict())
        else:
            raise(RuntimeError("type(src_polygon) = {}".format(type(src_polygon))))
    
    objects = []
    for idx, (dst_polygon, dst_property) in enumerate(zip(dst_polygons, dst_properties)):
        object_struct = dict()

        if clean_polygon_flag:
            if not dst_polygon.is_valid:
                continue
            if dst_polygon.geom_type not in ['Polygon', 'MultiPolygon']:
                continue

        object_struct['mask'] = bstool.polygon2mask(dst_polygon)
        xmin, ymin, xmax, ymax = dst_polygon.bounds
        object_struct['bbox'] = [xmin, ymin, xmax, ymax]
        object_struct['polygon'] = dst_polygon
        object_struct['property'] = dst_property

        objects.append(object_struct)

    return objects

def mask_parse(mask_file,
               subclasses=(1, 3),
               clean_polygon_flag=False,
               with_opencv=False):
    """parse mask image

    Args:
        mask_file (str or np.array): mask image
        subclasses (tuple, optional): parse which class. Defaults to (1, 3).

    Returns:
        list: list of objects
    """
    if isinstance(mask_file, str):
        mask_image = cv2.imread(mask_file)
    else:
        mask_image = mask_file

    if mask_image is None:
        if isinstance(mask_file, str):
            print("Can not open this mask (ignore) file: {}".format(mask_file))
        else:
            print("Can not handle mask image (np.array), it is empty")
        return []

    sub_mask = bstool.generate_subclass_mask(mask_image, subclasses=subclasses)
    if with_opencv:
        polygons = bstool.generate_polygon_opencv(sub_mask)
    else:
        polygons = bstool.generate_polygon(sub_mask)

    objects = []
    for polygon in polygons:
        object_struct = dict()
        if clean_polygon_flag:
            if not polygon.is_valid:
                continue
            if polygon.geom_type not in ['Polygon', 'MultiPolygon']:
                continue
        object_struct['mask'] = bstool.polygon2mask(polygon)
        if len(object_struct['mask']) == 0:
            continue
        object_struct['polygon'] = polygon
        objects.append(object_struct)

    return objects
        
def bs_json_parse(json_file):
    """parse json file

    Args:
        json_file (str): json file

    Returns:
        list: list of objects
    """
    annotations = mmcv.load(json_file)['annotations']
    objects = []
    for annotation in annotations:
        object_struct = {}
        roof_mask = annotation['roof']
        roof_polygon = bstool.mask2polygon(roof_mask)
        roof_bound = roof_polygon.bounds
        footprint_mask = annotation['footprint']
        footprint_polygon = bstool.mask2polygon(footprint_mask)
        footprint_bound = footprint_polygon.bounds
        building_xmin = np.minimum(roof_bound[0], footprint_bound[0])
        building_ymin = np.minimum(roof_bound[1], footprint_bound[1])
        building_xmax = np.maximum(roof_bound[2], footprint_bound[2])
        building_ymax = np.maximum(roof_bound[3], footprint_bound[3])

        building_bound = [building_xmin, building_ymin, building_xmax, building_ymax]

        xmin, ymin, xmax, ymax = list(roof_bound)
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
        object_struct['roof_bbox'] = object_struct['bbox']
        xmin, ymin, xmax, ymax = list(footprint_bound)
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        object_struct['footprint_bbox'] = [xmin, ymin, bbox_w, bbox_h]
        xmin, ymin, xmax, ymax = list(building_bound)
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        object_struct['building_bbox'] = [xmin, ymin, bbox_w, bbox_h]

        object_struct['roof_mask'] = roof_mask
        object_struct['footprint_mask'] = footprint_mask
        object_struct['ignore_flag'] = annotation['ignore']
        object_struct['offset'] = annotation['offset']
        object_struct['building_height'] = annotation['building_height']
        
        object_struct['segmentation'] = roof_mask
        object_struct['label'] = 1
        object_struct['iscrowd'] = object_struct['ignore_flag']
        
        objects.append(object_struct)
    
    return objects

def urban3d_json_parse(json_file):
    """parse json file

    Args:
        json_file (str): json file

    Returns:
        list: list of objects
    """
    annotations = mmcv.load(json_file)['annotations']
    objects = []
    for annotation in annotations:
        object_struct = {}
        roof_mask = annotation['roof']
        roof_polygon = bstool.mask2polygon(roof_mask)
        roof_bound = roof_polygon.bounds
        footprint_mask = annotation['footprint']
        footprint_polygon = bstool.mask2polygon(footprint_mask)
        footprint_bound = footprint_polygon.bounds
        building_xmin = np.minimum(roof_bound[0], footprint_bound[0])
        building_ymin = np.minimum(roof_bound[1], footprint_bound[1])
        building_xmax = np.maximum(roof_bound[2], footprint_bound[2])
        building_ymax = np.maximum(roof_bound[3], footprint_bound[3])

        building_bound = [building_xmin, building_ymin, building_xmax, building_ymax]

        xmin, ymin, xmax, ymax = list(roof_bound)
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
        object_struct['roof_bbox'] = object_struct['bbox']
        xmin, ymin, xmax, ymax = list(footprint_bound)
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        object_struct['footprint_bbox'] = [xmin, ymin, bbox_w, bbox_h]
        xmin, ymin, xmax, ymax = list(building_bound)
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        object_struct['building_bbox'] = [xmin, ymin, bbox_w, bbox_h]

        object_struct['roof_mask'] = roof_mask
        object_struct['footprint_mask'] = footprint_mask
        object_struct['offset'] = annotation['offset']
        object_struct['building_height'] = annotation['building_height']
        
        object_struct['segmentation'] = roof_mask
        object_struct['label'] = 1
        
        objects.append(object_struct)
    
    return objects


class COCOParse():
    def __init__(self, anno_file, classes=['']):
        self.anno_info = dict()
        self.coco =  COCO(anno_file)
        catIds = self.coco.getCatIds(catNms=classes)
        imgIds = self.coco.getImgIds(catIds=catIds)

        for idx, imgId in enumerate(imgIds):
            img = self.coco.loadImgs(imgIds[idx])[0]
            annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = self.coco.loadAnns(annIds)

            self.anno_info[img['file_name']] = anns

    def __call__(self, image_fn):
        return self.anno_info[image_fn]


class BSPklParser():
    def __init__(self, 
                 anno_file, 
                 pkl_file, 
                 iou_threshold=0.1,
                 score_threshold=0.05,
                 min_area=500,
                 with_offset=False,
                 with_height=False,
                 gt_roof_csv_file=None,
                 replace_pred_roof=False,
                 offset_model='footprint2roof'):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.min_area = min_area
        self.with_offset = with_offset
        self.with_height = with_height
        self.offset_model = offset_model

        self.replace_pred_roof = replace_pred_roof

        if not self.with_offset:
            if self.with_height:
                raise(RuntimeError('not support with_offset={}, with_height={}'.format(self.with_offset, self.with_height)))
        
        if isinstance(pkl_file, str):
            results = mmcv.load(pkl_file)
        else:
            results = pkl_file
            
        coco = COCO(anno_file)
        img_ids = coco.get_img_ids()

        self.objects = dict()
        self.building_with_coord = defaultdict(dict)
        for idx, img_id in tqdm.tqdm(enumerate(img_ids)):
            info = coco.load_imgs([img_id])[0]
            img_name = bstool.get_basename(info['file_name'])
            sub_fold, ori_image_name, coord = bstool.get_info_splitted_imagename(img_name)

            result = results[idx]

            self.building_with_coord[ori_image_name][coord] = self._convert_items(result)
            self.objects[img_name] = self.building_with_coord[ori_image_name][coord][:]
        
        print('========== begin to merge the polygons in pkl ==========')
        self.merged_objects = self._merge_buildings()

        if self.replace_pred_roof:
            print('========== replace roof prediction with groundtruth ==========')
            self._replace_pred_roof_with_gt_roof(gt_roof_csv_file)

        print("Finish init the pkl parser")

    def _replace_pred_roof_with_gt_roof(self, gt_roof_csv_file):
        gt_csv_parser = bstool.CSVParse(gt_roof_csv_file)
        gt_objects = gt_csv_parser.objects
        ori_image_name_list = gt_csv_parser.image_name_list

        for ori_image_name in self.ori_image_name_list:
            buildings = dict()
            
            gt_buildings = gt_objects[ori_image_name]
            pred_buildings = self.merged_objects[ori_image_name]

            gt_polygons = [gt_building['polygon'] for gt_building in gt_buildings]
            pred_polygons = [building['roof_polygon'] for building in pred_buildings]
            offsets = [building['offset'] for building in pred_buildings]

            gt_polygons_origin = gt_polygons[:]

            if len(gt_polygons) == 0 or len(pred_polygons) == 0:
                continue

            gt_polygons = geopandas.GeoSeries(gt_polygons)
            pred_polygons = geopandas.GeoSeries(pred_polygons)

            gt_df = geopandas.GeoDataFrame({'geometry': gt_polygons, 'gt_df':range(len(gt_polygons))})
            pred_df = geopandas.GeoDataFrame({'geometry': pred_polygons, 'pred_df':range(len(pred_polygons))})

            res_intersection = geopandas.overlay(gt_df, pred_df, how='intersection')

            iou = np.zeros((len(pred_polygons), len(gt_polygons)))
            for idx, row in res_intersection.iterrows():
                gt_idx = row.gt_df
                pred_idx = row.pred_df

                inter = row.geometry.area
                union = pred_polygons[pred_idx].area + gt_polygons[gt_idx].area

                iou[pred_idx, gt_idx] = inter / (union - inter + 1.0)

            replace_index = np.argmax(iou, axis=-1)
            new_pred_polygons = np.array(gt_polygons_origin)[replace_index].tolist()

            buildings = self.merged_objects[ori_image_name]
            num_buildings = len(buildings)
            for index in range(num_buildings):
                self.merged_objects[ori_image_name][index]['roof_polygon'] = new_pred_polygons[index]
            
            for idx, (offset, roof_polygon) in enumerate(zip(offsets, new_pred_polygons)):
                transform_matrix = [1, 0, 0, 1,  -1.0 * offset[0], -1.0 * offset[1]]
                footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)

                self.merged_objects[ori_image_name][idx]['footprint_polygon'] = footprint_polygon

    def _merge_buildings(self):
        self.ori_image_name_list = list(self.building_with_coord.keys())

        merged_objects = dict()
        for ori_image_name in self.ori_image_name_list:
            subimage_coordinates = self.building_with_coord[ori_image_name]

            merged_buildings = []
            polygons_merged, scores_merged = [], []
            for subimage_coordinate in subimage_coordinates:
                buildings = self.building_with_coord[ori_image_name][subimage_coordinate]

                if len(buildings) == 0:
                    continue

                buildings = self._chang_building_coordinate(buildings, subimage_coordinate)
                merged_buildings += buildings

                footprint_polygons = [building['footprint_polygon'] for building in buildings]
                scores = [building['score'] for building in buildings]

                polygons_merged += footprint_polygons
                scores_merged += scores

            keep = self._mask_nms(polygons_merged, np.array(scores_merged), iou_threshold=self.iou_threshold)

            merged_objects[ori_image_name] = np.array(merged_buildings)[keep].tolist()

        return merged_objects

    def _chang_building_coordinate(self, buildings, coordinate):
        transform_matrix = [1, 0, 0, 1, coordinate[0], coordinate[1]]
        
        transformed_buildings = []
        for building in buildings:
            roof_polygon = building['roof_polygon']
            footprint_polygon = building['footprint_polygon']

            transformed_roof_polygon = affinity.affine_transform(roof_polygon, transform_matrix)
            transformed_footprint_polygon = affinity.affine_transform(footprint_polygon, transform_matrix)
            building['roof_polygon'] = transformed_roof_polygon
            building['footprint_polygon'] = transformed_footprint_polygon

            transformed_buildings.append(building)

        return transformed_buildings

    def _mask_nms(self, masks, scores, iou_threshold=0.5):
        """non-maximum suppression (NMS) on the masks according to their intersection-over-union (IoU)
        
        Arguments:
            masks {np.array} -- [N * 4]
            scores {np.array} -- [N * 1]
            iou_threshold {float} -- threshold for IoU
        """
        polygons = np.array(masks)

        areas = np.array([polygon.area for polygon in polygons])

        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            best_mask_idx = order[0]
            keep.append(best_mask_idx)

            best_mask = polygons[best_mask_idx]
            remain_masks = polygons[order[1:]]

            inters = []
            for remain_mask in remain_masks:
                mask1 = best_mask
                mask2 = remain_mask
                try:
                    inter = mask1.intersection(mask2).area
                except:
                    inter = 2048 * 2048
                inters.append(inter)

            inters = np.array(inters)
            iou = inters / (areas[best_mask_idx] + areas[order[1:]] - inters)

            inds = np.where(iou <= iou_threshold)[0]
            
            order = order[inds + 1]

        return keep

    def _convert_items(self, result):
        buildings = []
        if self.with_offset and not self.with_height:
            det, seg, offset = result
            if isinstance(offset, list):
                return []
            else:
                height = np.zeros((offset.shape[0], 1))
        if self.with_height:
            det, seg, offset, height = result

        bboxes = np.vstack(det)
        segms = mmcv.concat_list(seg)

        if isinstance(offset, tuple):
            offsets = offset[0]
        else:
            offsets = offset

        if self.with_height and isinstance(height, tuple):
            heights = height[0]
        else:
            heights = height

        for i in range(bboxes.shape[0]):
            building = dict()
            score = bboxes[i][4]
            if score < self.score_threshold:
                continue

            if isinstance(segms[i]['counts'], bytes):
                segms[i]['counts'] = segms[i]['counts'].decode()
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            gray = np.array(mask * 255, dtype=np.uint8)

            contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            
            if contours != []:
                cnt = max(contours, key = cv2.contourArea)
                if cv2.contourArea(cnt) < 5:
                    continue
                mask = np.array(cnt).reshape(1, -1).tolist()[0]
                if len(mask) < 8:
                    continue

                valid_flag = bstool.single_valid_polygon(bstool.mask2polygon(mask))
                if not valid_flag:
                    continue
            else:
                continue

            bbox = bboxes[i][0:4]
            offset = offsets[i]
            height = heights[i][0]
            roof = mask

            roof_polygon = bstool.mask2polygon(roof)

            if roof_polygon.area < self.min_area:
                continue
            
            if self.offset_model == 'footprint2roof':
                transform_matrix = [1, 0, 0, 1,  -1.0 * offset[0], -1.0 * offset[1]]
            elif self.offset_model == 'roof2footprint':
                transform_matrix = [1, 0, 0, 1,  1.0 * offset[0], 1.0 * offset[1]]
            else:
                raise NotImplementedError
            footprint_polygon = affinity.affine_transform(roof_polygon, transform_matrix)

            building['bbox'] = bbox.tolist()
            building['offset'] = offset.tolist()
            building['height'] = height
            building['score'] = score
            building['roof_polygon'] = roof_polygon
            building['footprint_polygon'] = footprint_polygon

            buildings.append(building)
        
        return buildings

    def __call__(self, image_fn, splitted=False):
        if splitted:
            if image_fn in self.objects.keys():
                return self.objects[image_fn]
            else:
                print("{} is not in pkl".format(image_fn))
                return []
        else:
            if image_fn in self.merged_objects.keys():
                return self.merged_objects[image_fn]
            else:
                print("{} is not in pkl".format(image_fn))
                return []

class BSPklParser_Only_Offset(BSPklParser):
    def _convert_items(self, result):
        buildings = []
        det, offset = result

        bboxes = np.vstack(det)

        if isinstance(offset, tuple):
            offsets = offset[0]
        else:
            offsets = offset

        for i in range(bboxes.shape[0]):
            building = dict()
            score = bboxes[i][4]
            if score < self.score_threshold:
                continue
        
            bbox = bboxes[i][0:4]
            bbox = bbox.tolist()
            offset = offsets[i]

            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            if w < 2 or h < 2:
                continue

            if bbox[0] < 0 or bbox[1] > 1023 or bbox[2] < 0 or bbox[3] > 1023:
                continue

            roof = bstool.bbox2pointobb(bbox)   # fake roof

            roof_polygon = bstool.mask2polygon(roof)

            valid_flag = bstool.single_valid_polygon(roof_polygon)
            if not valid_flag:
                continue

            if roof_polygon.area < self.min_area:
                continue

            building['bbox'] = bbox
            building['offset'] = offset.tolist()
            building['height'] = 0.0
            building['score'] = score
            building['roof_polygon'] = roof_polygon
            building['footprint_polygon'] = roof_polygon

            buildings.append(building)
        
        return buildings

class CSVParse():
    def __init__(self, csv_file, min_area=0, check_valid=True):
        csv_df = pandas.read_csv(csv_file)
        self.image_name_list = list(set(csv_df.ImageId.unique()))

        self.objects = defaultdict(dict)
        self.dataset_buildings = []
        print("begin parsing the csv file")
        for image_name in tqdm.tqdm(self.image_name_list):
            buildings = []
            for idx, row in csv_df[csv_df.ImageId == image_name].iterrows():
                building = dict()
                obj_keys = row.to_dict().keys()
                polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
                if polygon.area < min_area:
                    continue
                building['polygon'] = polygon
                building['mask'] = bstool.polygon2mask(polygon)

                if check_valid and not bstool.single_valid_polygon(building['polygon']):
                    # building['polygon'] = building['polygon'].buffer(0.0)
                    continue
                
                building['score'] = row.Confidence
                if 'Offset' in obj_keys:
                    if type(row.Offset) == str:
                        if row.Offset == '[0 0]':
                            building['offset'] = [0, 0]
                        else:
                            building['offset'] = ast.literal_eval(row.Offset)
                    else:
                        building['offset'] = row.Offset
                else:
                    building['offset'] = [0, 0]

                if 'Height' in obj_keys:
                    building['height'] = row.Height
                else:
                    building['height'] = 0

                buildings.append(building)

            self.objects[image_name] = buildings
            self.dataset_buildings += buildings

    def __call__(self, image_fn):
        if image_fn in self.objects.keys():
            return self.objects[image_fn]
        else:
            print("{} is not in csv file".format(image_fn))
            return []