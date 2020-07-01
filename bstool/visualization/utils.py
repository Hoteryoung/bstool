import numpy as np
import cv2
import pandas
import geopandas
import shapely
import matplotlib.pyplot as plt


def draw_grid(img, line_color=(0, 255, 0), thickness=2, type_=cv2.LINE_AA, pxstep=512):
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

    return img

def get_confusion_matrix_indexes(pred_csv_file, gt_csv_file, show_matplotlib=False):
    pred_csv_df = pandas.read_csv(pred_csv_file)
    gt_csv_df = pandas.read_csv(gt_csv_file)

    image_name_list = list(set(gt_csv_df.ImageId.unique()))
    gt_TP_indexes, pred_TP_indexes, gt_FN_indexes, pred_FP_indexes = dict(), dict(), dict(), dict()
    dataset_gt_polygons, dataset_pred_polygons = dict(), dict()
    for image_name in    image_name_list:
        gt_polygons, pred_polygons = [], []
        for idx, row in gt_csv_df[gt_csv_df.ImageId == image_name].iterrows():
            gt_polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
            if not gt_polygon.is_valid:
                continue
            gt_polygons.append(gt_polygon)

        for idx, row in pred_csv_df[pred_csv_df.ImageId == image_name].iterrows():
            pred_polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
            if not pred_polygon.is_valid:
                continue
            pred_polygons.append(pred_polygon)

        gt_polygons = geopandas.GeoSeries(gt_polygons)
        pred_polygons = geopandas.GeoSeries(pred_polygons)

        dataset_gt_polygons[image_name] = gt_polygons
        dataset_pred_polygons[image_name] = pred_polygons

        gt_df = geopandas.GeoDataFrame({'geometry': gt_polygons, 'gt_df':range(len(gt_polygons))})
        pred_df = geopandas.GeoDataFrame({'geometry': pred_polygons, 'pred_df':range(len(pred_polygons))})

        res_intersection = geopandas.overlay(gt_df, pred_df, how='intersection')

        iou = np.zeros((len(pred_polygons), len(gt_polygons)))
        for idx, row in res_intersection.iterrows():
            gt_idx = row.gt_df
            pred_idx = row.pred_df

            inter = row.geometry.area

            union = pred_polygons[pred_idx].area + gt_polygons[gt_idx].area

            iou[pred_idx, gt_idx] = inter / (union - inter)
        
        iou_indexes = np.argwhere(iou >= 0.5)

        gt_TP_indexes[image_name] = list(iou_indexes[:, 1])
        pred_TP_indexes[image_name] = list(iou_indexes[:, 0])

        gt_FN_indexes[image_name] = list(set(range(len(gt_polygons))) - set(gt_TP_indexes[image_name]))
        pred_FP_indexes[image_name] = list(set(range(len(pred_polygons))) - set(pred_TP_indexes[image_name]))

        if show_matplotlib:
            fig, ax = plt.subplots(1, 1)

            gt_df.plot(ax=ax, color='red')
            pred_df.plot(ax=ax, facecolor='none', edgecolor='k')

            plt.xlim(0, 2048)
            plt.ylim(0, 2048)
            plt.gca().invert_yaxis()

            plt.show()

    return gt_TP_indexes, pred_TP_indexes, gt_FN_indexes, pred_FP_indexes, dataset_gt_polygons, dataset_pred_polygons