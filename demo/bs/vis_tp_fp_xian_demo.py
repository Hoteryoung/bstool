import os
import cv2
import bstool
import mmcv


if __name__ == '__main__':
    model = 'bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0'

    # RGB
    colors = {'gt_TP':   (0, 255, 0),
              'pred_TP': (255, 255, 0),
              'FP':      (0, 255, 255),
              'FN':      (255, 0, 0)}

    for mask_type in ['footprint', 'roof']:
        gt_csv_file = f'/data/buildchange/v0/xian_fine/xian_val_{mask_type}_gt_minarea100.csv'
        pred_csv_file = f'/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/buildchange/bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0/bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0_iou_threshold_0.1_score_threshold_0.4_min_area_500_epoch_12_xian_footprint_merged.csv'

        image_dir = '/data/buildchange/v0/xian_fine/images'
        output_dir = f'/data/buildchange/v0/xian_fine/vis/{model}_{mask_type}'
        bstool.mkdir_or_exist(output_dir)

        gt_TP_indexes, pred_TP_indexes, gt_FN_indexes, pred_FP_indexes, dataset_gt_polygons, dataset_pred_polygons, gt_ious = bstool.get_confusion_matrix_indexes(pred_csv_file, gt_csv_file)

        confusion_matrix = [gt_TP_indexes, pred_TP_indexes, gt_FN_indexes, pred_FP_indexes]

        TP_dataset, FN_dataset, FP_dataset = 0, 0, 0
        progress_bar = mmcv.ProgressBar(len(os.listdir(image_dir)))
        for image_name in os.listdir(image_dir):
            image_basename = bstool.get_basename(image_name)
            image_file = os.path.join(image_dir, image_name)

            output_file = os.path.join(output_dir, image_name)

            img = cv2.imread(image_file)

            TP = len(gt_TP_indexes[image_basename])
            FN = len(gt_FN_indexes[image_basename])
            FP = len(pred_FP_indexes[image_basename])

            TP_dataset += TP
            FN_dataset += FN
            FP_dataset += FP

            for idx, gt_polygon in enumerate(dataset_gt_polygons[image_basename]):
                iou = gt_ious[image_basename][idx]

                if idx in gt_TP_indexes[image_basename]:
                    color = colors['gt_TP'][::-1]
                else:
                    color = colors['FN'][::-1]

                img = bstool.draw_mask_boundary(img, bstool.polygon2mask(gt_polygon), color=color)
                img = bstool.draw_iou(img, gt_polygon, iou, color=color)

            for idx, pred_polygon in enumerate(dataset_pred_polygons[image_basename]):
                if idx in pred_TP_indexes[image_basename]:
                    color = colors['pred_TP'][::-1]
                else:
                    color = colors['FP'][::-1]

                img = bstool.draw_mask_boundary(img, bstool.polygon2mask(pred_polygon), color=color)
            
            img = bstool.draw_confusion_matrix_on_image(img, image_basename, confusion_matrix)
            # bstool.show_image(img, output_file=output_file)
            cv2.imwrite(output_file, img)
            progress_bar.update()
        
        print("TP, FN, FP: ", TP_dataset, FN_dataset, FP_dataset)