import bstool


if __name__ == "__main__":
    model = 'bc_v015_mask_rcnn_r50_v2_roof_trainval'
    image_set = 'dalian_fine'

    pred_csv_file = f'/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/{model}/{model}_{image_set}_roof.csv'
    gt_csv_file = f'/data/buildchange/v0/xian_fine/xian_val_footprint_gt_minarea100.csv'

    merged_csv_file = pred_csv_file.split('.csv')[0] + f"_merged.csv"
    merged_csv_file = '/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/buildchange/bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0/bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0_iou_threshold_0.1_score_threshold_0.4_min_area_500_epoch_12_xian_footprint_merged.csv'
    # bstool.merge_csv_results(pred_csv_file, merged_csv_file, iou_threshold=0.1, score_threshold=0.50, min_area=100)

    bstool.solaris_semantic_evaluation(merged_csv_file, gt_csv_file)