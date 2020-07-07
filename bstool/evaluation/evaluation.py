import os
import cv2
import numpy as np
import geopandas

import bstool


class Evaluation():
    def __init__(self,
                 model=None,
                 anno_file=None,
                 pkl_file=None,
                 gt_roof_csv_file=None,
                 gt_footprint_csv_file=None,
                 roof_csv_file=None,
                 rootprint_csv_file=None,
                 iou_threshold=0.1,
                 score_threshold=0.4,
                 with_offset=False,
                 with_height=False):
        self.gt_roof_csv_file = gt_roof_csv_file
        self.gt_footprint_csv_file = gt_footprint_csv_file
        self.roof_csv_file = roof_csv_file
        self.rootprint_csv_file = rootprint_csv_file

        self.pkl_file = pkl_file

        pkl_parser = bstool.BSPklParser(anno_file, 
                                        pkl_file, 
                                        iou_threshold=iou_threshold, 
                                        score_threshold=score_threshold, 
                                        with_offset=with_offset, 
                                        with_height=with_height)

        merged_objects = pkl_parser.merged_objects
        
        bstool.bs_csv_dump(merged_objects, roof_csv_file, rootprint_csv_file)

    def segmentation(self):
        bstool.solaris_semantic_evaluation(self.roof_csv_file, self.gt_roof_csv_file)
        bstool.solaris_semantic_evaluation(self.rootprint_csv_file, self.gt_footprint_csv_file)

    def detection(self):
        pass

    def height(self):
        objects = self.get_confusion_matrix_indexes()

        errors = []
        gt_angle_std = []
        pred_angle_std = []
        for ori_image_name in self.ori_image_name_list:
            if ori_image_name not in objects.keys():
                continue

            gt_heights = objects[ori_image_name]['gt_heights']
            pred_heights = objects[ori_image_name]['pred_heights']

            if len(gt_heights) == 0 or len(pred_heights) == 0:
                continue

            gt_offsets = np.array(objects[ori_image_name]['gt_offsets'])
            pred_offsets = np.array(objects[ori_image_name]['pred_offsets'])

            error = np.abs(np.array(gt_heights) - np.array(pred_heights))
            errors += error.tolist()

            # gt_avg_offsets = np.array(gt_offsets) / np.array(gt_heights)
            gt_angle = np.arctan2(gt_offsets[:, 1], gt_offsets[:, 0])

            # pred_avg_offsets = np.array(pred_offsets) / np.array(pred_heights)
            pred_angle = np.arctan2(pred_offsets[:, 1], pred_offsets[:, 0])

            single_image_gt_angle_std = gt_angle.std()
            single_image_pred_angle_std = pred_angle.std()

            gt_angle_std.append(single_image_gt_angle_std)
            pred_angle_std.append(single_image_pred_angle_std)

        mse = np.array(errors).mean()

        print("Evaluation pkl file: ", self.pkl_file)
        print("Height MSE: ", mse)
        print("GT angle Std: ", np.array(gt_angle_std).mean())
        print("Pred angle Std: ", np.array(pred_angle_std).mean())

    def get_confusion_matrix_indexes(self):
        gt_csv_parser = bstool.CSVParse(self.gt_footprint_csv_file)
        pred_csv_parser = bstool.CSVParse(self.rootprint_csv_file)

        self.ori_image_name_list = gt_csv_parser.image_name_list

        gt_objects = gt_csv_parser.objects
        pred_objects = pred_csv_parser.objects

        objects = dict()

        for ori_image_name in self.ori_image_name_list:
            buildings = dict()
            
            gt_buildings = gt_objects[ori_image_name]
            pred_buildings = pred_objects[ori_image_name]

            gt_polygons = [gt_building['polygon'] for gt_building in gt_buildings]
            pred_polygons = [pred_building['polygon'] for pred_building in pred_buildings]
            
            if len(gt_polygons) == 0 or len(pred_polygons) == 0:
                continue

            gt_offsets = [gt_building['offset'] for gt_building in gt_buildings]
            pred_offsets = [pred_building['offset'] for pred_building in pred_buildings]
            
            gt_heights = [gt_building['height'] for gt_building in gt_buildings]
            pred_heights = [pred_building['height'] for pred_building in pred_buildings]

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

            iou_indexes = np.argwhere(iou >= 0.5)

            gt_TP_indexes = list(iou_indexes[:, 1])
            pred_TP_indexes = list(iou_indexes[:, 0])

            gt_FN_indexes = list(set(range(len(gt_polygons))) - set(gt_TP_indexes))
            pred_FP_indexes = list(set(range(len(pred_polygons))) - set(pred_TP_indexes))

            buildings['gt_iou'] = np.max(iou, axis=0)

            buildings['gt_TP_indexes'] = gt_TP_indexes
            buildings['pred_TP_indexes'] = pred_TP_indexes
            buildings['gt_FN_indexes'] = gt_FN_indexes
            buildings['pred_FP_indexes'] = pred_FP_indexes

            buildings['gt_offsets'] = np.array(gt_offsets)[gt_TP_indexes].tolist()
            buildings['pred_offsets'] = np.array(pred_offsets)[pred_TP_indexes].tolist()

            buildings['gt_heights'] = np.array(gt_heights)[gt_TP_indexes].tolist()
            buildings['pred_heights'] = np.array(pred_heights)[pred_TP_indexes].tolist()

            objects[ori_image_name] = buildings

        return objects