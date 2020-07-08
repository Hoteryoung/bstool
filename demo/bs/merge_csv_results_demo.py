import bstool


if __name__ == "__main__":
    model = 'bc_v015_mask_rcnn_r50_v2_roof_trainval'

    pred_csv_file = f'/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/{model}/{model}_footprint.csv'
    gt_csv_file = f'/data/buildchange/v0/xian_fine/xian_val_footprint_gt_minarea100.csv'

    for score in [0.6, 0.65, 0.1, 0.15]:
        print("score: ", score)
        merged_csv_file = pred_csv_file.split('.csv')[0] + f"_merged_{score}.csv"
        bstool.merge_csv_results(pred_csv_file, merged_csv_file, iou_threshold=0.1, score_threshold=score, min_area=100)
