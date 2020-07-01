import os
import cv2
import bstool


if __name__ == '__main__':
    pred_csv_file = '/data/buildchange/v0/xian_fine/evaluation_results/bc_v015_mask_rcnn_r50_v2_roof_trainval_roof_merged.csv'
    gt_csv_file = '/data/buildchange/v0/xian_fine/xian_val_roof_gt_minarea100.csv'
    image_dir = '/data/buildchange/v0/xian_fine/images'
    output_dir = '/data/buildchange/v0/xian_fine/vis/v015_roof'
    bstool.mkdir_or_exist(output_dir)

    # RGB
    colors = {'gt_TP':   (0, 255, 0),
              'pred_TP': (255, 255, 0),
              'FP':      (0, 255, 255),
              'FN':      (255, 0, 0)}

    gt_TP_indexes, pred_TP_indexes, gt_FN_indexes, pred_FP_indexes, dataset_gt_polygons, dataset_pred_polygons = bstool.get_confusion_matrix_indexes(pred_csv_file, gt_csv_file)

    for image_name in os.listdir(image_dir):
        image_basename = bstool.get_basename(image_name)
        image_file = os.path.join(image_dir, image_name)

        output_file = os.path.join(output_dir, image_name)

        img = cv2.imread(image_file)

        for idx, gt_polygon in enumerate(dataset_gt_polygons[image_basename]):
            if idx in gt_TP_indexes[image_basename]:
                color = colors['gt_TP'][::-1]
            else:
                color = colors['FN'][::-1]

            img = bstool.draw_mask_boundary(img, bstool.polygon2mask(gt_polygon), color=color)

        for idx, pred_polygon in enumerate(dataset_pred_polygons[image_basename]):
            if idx in pred_TP_indexes[image_basename]:
                color = colors['pred_TP'][::-1]
            else:
                color = colors['FP'][::-1]

            img = bstool.draw_mask_boundary(img, bstool.polygon2mask(pred_polygon), color=color)
        
        bstool.show_image(img, output_file=output_file)