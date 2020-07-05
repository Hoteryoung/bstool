import os
import numpy as np
import cv2
import pandas
import geopandas
import shapely
import mmcv
import bstool
import math


def get_confusion_matrix_indexes(pred_csv_file, gt_csv_file, show_matplotlib=False):
    pred_csv_df = pandas.read_csv(pred_csv_file)
    gt_csv_df = pandas.read_csv(gt_csv_file)

    image_name_list = list(set(gt_csv_df.ImageId.unique()))
    gt_TP_indexes, pred_TP_indexes, gt_FN_indexes, pred_FP_indexes = dict(), dict(), dict(), dict()
    dataset_gt_polygons, dataset_gt_heights, dataset_pred_polygons, dataset_pred_heights = dict(), dict(), dict(), dict()
    gt_ious, dataset_ious = dict(), dict()
    for image_name in    image_name_list:
        gt_polygons, gt_heights, pred_polygons, pred_heights = [], [], [], []
        for idx, row in gt_csv_df[gt_csv_df.ImageId == image_name].iterrows():
            gt_polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
            gt_height = float(row.BuildingHeight)
            if not gt_polygon.is_valid:
                continue
            gt_polygons.append(gt_polygon)
            gt_heights.append(gt_height)

        for idx, row in pred_csv_df[pred_csv_df.ImageId == image_name].iterrows():
            pred_polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
            pred_height = float(row.BuildingHeight)
            if not pred_polygon.is_valid:
                continue
            pred_polygons.append(pred_polygon)
            pred_heights.append(pred_height)

        if len(gt_polygons) == 0 or len(pred_polygons) == 0:
            continue

        gt_polygons = geopandas.GeoSeries(gt_polygons)
        pred_polygons = geopandas.GeoSeries(pred_polygons)

        dataset_gt_polygons[image_name] = gt_polygons
        dataset_gt_heights[image_name] = gt_heights
        dataset_pred_polygons[image_name] = pred_polygons
        dataset_pred_heights[image_name] = pred_heights

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

        gt_ious[image_name] = np.max(iou, axis=0)
        dataset_ious[image_name] = iou

        gt_TP_indexes[image_name] = list(iou_indexes[:, 1])
        pred_TP_indexes[image_name] = list(iou_indexes[:, 0])

        gt_FN_indexes[image_name] = list(set(range(len(gt_polygons))) - set(gt_TP_indexes[image_name]))
        pred_FP_indexes[image_name] = list(set(range(len(pred_polygons))) - set(pred_TP_indexes[image_name]))

    return gt_TP_indexes, pred_TP_indexes, gt_FN_indexes, pred_FP_indexes, dataset_gt_polygons, dataset_gt_heights, dataset_pred_polygons, dataset_pred_heights, gt_ious, dataset_ious


if __name__ == '__main__':
    pred_csv_file = '/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/buildchange/bc_v006_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox/bc_v006_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_dalian_coco_results_footprint_merged.csv'
    gt_csv_file = '/data/buildchange/v0/dalian_fine/dalian_fine_2048_footprint_gt.csv'
    image_dir = '/data/buildchange/v0/dalian_fine/images'
    output_dir = '/data/buildchange/v0/dalian_fine/vis/v015_footprint'
    bstool.mkdir_or_exist(output_dir)

    # RGB
    colors = {'gt_TP':   (0, 255, 0),
              'pred_TP': (255, 255, 0),
              'FP':      (0, 255, 255),
              'FN':      (255, 0, 0)}

    gt_TP_indexes, pred_TP_indexes, gt_FN_indexes, pred_FP_indexes, dataset_gt_polygons, dataset_gt_heights, dataset_pred_polygons, dataset_pred_heights, gt_ious, dataset_ious = get_confusion_matrix_indexes(pred_csv_file, gt_csv_file)

    gt_tp_heights, pred_tp_heights = [], []
    errors = []

    for image_name in os.listdir(image_dir):
        image_basename = bstool.get_basename(image_name)
        image_file = os.path.join(image_dir, image_name)

        output_file = os.path.join(output_dir, image_name)

        img = cv2.imread(image_file)
        
        if image_basename not in dataset_gt_polygons or image_basename not in dataset_pred_polygons:
            continue

        for gt_idx, pred_idx in zip(gt_TP_indexes[image_basename], pred_TP_indexes[image_basename]):
            gt_height = dataset_gt_heights[image_basename][gt_idx]
            pred_height = dataset_pred_heights[image_basename][pred_idx]

            iou = dataset_ious[image_basename][pred_idx, gt_idx]

            print(gt_height, pred_height, iou)

            errors.append(abs(gt_height/3.0 - pred_height))

    print("Errors: ", np.array(errors).mean())
    print("Errors: ", np.array(errors).std())