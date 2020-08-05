import bstool


if __name__ == '__main__':
    # models = ['bc_v005.08.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin', 'bc_v005.08.03_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin_no_norm']
    models = ['bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0']
    # models = ['bc_v006.05_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_angle']
    # models = ['bc_v006.01_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_linear_50_50']
    # cities = ['jinan', 'shanghai', 'beijing','chengdu', 'haerbin']
    # cities = ['jinan', 'shanghai', 'beijing','chengdu', 'haerbin']
    cities = ['dalian', 'xian', 'xian_fixed']
    cities = ['urban3d']

    with_only_vis = True

    for model in models:
        version = model.split('_')[1]
        if 'v006' in model:
            with_height = True
        else:
            with_height = False
        for city in cities:
            print(f"========== {model} ========== {city} ==========")

            output_dir = f'./data/buildchange/v0/statistic/models/{model}'
            vis_dir = f'./data/buildchange/vis/{model}/{city}'
            bstool.mkdir_or_exist(vis_dir)
            image_dir = f'./data/buildchange/v0/{city}_fine/images'

            if city == 'xian':
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}_fine.json'
                gt_roof_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt.csv'
                gt_footprint_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_footprint_gt.csv'
            elif city == 'xian_fixed':
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_xian_fine.json'
                gt_roof_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt_fixed.csv'
                gt_footprint_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_footprint_gt_fixed.csv'
            elif city == 'dalian':
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}_fine.json'
                gt_roof_csv_file = f'./data/buildchange/v0/dalian_fine/dalian_fine_2048_roof_gt.csv'
                gt_footprint_csv_file = f'./data/buildchange/v0/dalian_fine/dalian_fine_2048_footprint_gt.csv'
            elif city == 'urban3d':
                imageset = 'val'
                anno_file = f'./data/urban3d/v1/coco/annotations/buildchange_v1_{imageset}_xian_fine.json'
                gt_roof_csv_file = './data/urban3d/v0/xian_fine/xian_fine_2048_roof_gt_fixed.csv'
                gt_footprint_csv_file = './data/urban3d/v0/xian_fine/xian_fine_2048_footprint_gt_fixed.csv'
            else:
                imageset = 'train'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}.json'
                gt_roof_csv_file = f'./data/buildchange/v0/{city}/{city}_2048_roof_gt.csv'
                gt_footprint_csv_file = f'./data/buildchange/v0/{city}/{city}_2048_footprint_gt.csv'
                

            if 'xian' in city:
                pkl_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_xian_coco_results.pkl'
            elif 'dalian' in city:
                pkl_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_dalian_coco_results.pkl'
            else:
                pkl_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_{city}_coco_results.pkl'
            
            roof_csv_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_roof_merged.csv'
            rootprint_csv_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_footprint_merged.csv'


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

            title = city + version
            if with_only_vis is False:
                evaluation.segmentation()
                evaluation.offset_length_classification(title=title)
                evaluation.offset_angle_classification(title=title)
                evaluation.offset_error_vector(title=title)
                if city == 'xian':
                    continue
                evaluation.height(percent=100, title=title)
                evaluation.visualization(image_dir=image_dir, vis_dir=vis_dir)
            else:
                evaluation.visualization(image_dir=image_dir, vis_dir=vis_dir)