import bstool


if __name__ == '__main__':
    # models = ['bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0', 'bc_v005.04_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_weight_2.0', 'bc_v005.06_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10', 'bc_v005.06.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10', 'bc_v005.07_offset_rcnn_r50_2x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10', 'bc_v006.02_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_log_mean_0_std_50', 'bc_v006.03_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_linear_21_24', 'bc_v006.04_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_share_conv', 'bc_v006.05_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_angle', 'bc_v006.06_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_share_conv_coupling']
    models = ['bc_v006.01_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_linear_50_50']
    # models = ['bc_v006.01_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_linear_50_50']
    # cities = ['jinan', 'shanghai', 'beijing','chengdu', 'haerbin']
    # cities = ['jinan', 'shanghai', 'beijing','chengdu', 'haerbin']
    cities = ['xian', 'dalian']

    for model in models:
        if 'v006' in model:
            with_height = True
        else:
            with_height = False
        for city in cities:
            print(f"========== {model} ========== {city} ==========")

            output_dir = f'./data/buildchange/v0/statistic/models/{model}'

            if 'xian' in city:
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}_fine.json'
                gt_roof_csv_file = f'./data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt.csv'
                gt_footprint_csv_file = f'./data/buildchange/v0/xian_fine/xian_fine_2048_footprint_gt.csv'
            elif 'dalian' in city:
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}_fine.json'
                gt_roof_csv_file = f'./data/buildchange/v0/dalian_fine/dalian_fine_2048_roof_gt.csv'
                gt_footprint_csv_file = f'./data/buildchange/v0/dalian_fine/dalian_fine_2048_footprint_gt.csv'
            else:
                imageset = 'train'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}.json'
                gt_roof_csv_file = f'./data/buildchange/v0/{city}/{city}_2048_roof_gt.csv'
                gt_footprint_csv_file = f'./data/buildchange/v0/{city}/{city}_2048_footprint_gt.csv'

            pkl_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_{city}_coco_results.pkl'
            
            roof_csv_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_roof_merged.csv'
            rootprint_csv_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_footprint_merged.csv'

            try:
                evaluation = bstool.Evaluation(model=model,
                                            anno_file=anno_file,
                                            pkl_file=pkl_file,
                                            gt_roof_csv_file=gt_roof_csv_file,
                                            gt_footprint_csv_file=gt_footprint_csv_file,
                                            roof_csv_file=roof_csv_file,
                                            rootprint_csv_file=rootprint_csv_file,
                                            iou_threshold=0.1,
                                            score_threshold=0.4,
                                            output_dir=output_dir,
                                            with_offset=True,
                                            with_height=with_height,
                                            show=True)

                evaluation.segmentation()
                evaluation.offset(title=city)
                if city == 'xian':
                    continue
                evaluation.height(percent=100, title=city)
            except:
                print(f"Error")