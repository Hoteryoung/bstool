import bstool


if __name__ == "__main__":
    model = 'bc_v015_mask_rcnn_r50_v2_roof_trainval'
    image_set = 'dalian_fine'

    pred_csv_file = f'/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/{model}/{model}_{image_set}_roof.csv'
    gt_csv_file = f'/data/buildchange/v0/dalian_fine/dalian_roof_gt_minarea100.csv'

    merged_csv_file = pred_csv_file.split('.csv')[0] + f"_merged.csv"
    bstool.merge_csv_results(pred_csv_file, merged_csv_file, iou_threshold=0.1, score_threshold=0.50, min_area=100)

    bstool.solaris_semantic_evaluation(merged_csv_file, gt_csv_file)