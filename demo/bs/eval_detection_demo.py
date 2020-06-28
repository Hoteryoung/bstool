import bstool


if __name__ == '__main__':
    pkl_file = '/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/buildchange/bc_v000_mask_rcnn_r50_1x_debug/bc_v000_mask_rcnn_r50_1x_debug_coco_results_merged.pkl'
    ann_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_train_samples_origin.json'
    json_prefix = 'bc_v000_mask_rcnn_r50_1x_debug'
    
    det_eval = bstool.DetEval(pkl_file, ann_file, json_prefix)
    det_eval.evaluation()