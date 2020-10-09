import bstool
import csv

def write_results2csv(results, meta_info=None):
    print("meta_info: ", meta_info)
    segmentation_eval_results, offset_eval_results, angle_eval_results, error_vector_results = results
    with open(meta_info['summary_file'], 'w') as summary:
        csv_writer = csv.writer(summary, delimiter=',')
        csv_writer.writerow(['Meta Info'])
        csv_writer.writerow(['model', meta_info['model']])
        csv_writer.writerow(['anno_file', meta_info['anno_file']])
        csv_writer.writerow(['gt_roof_csv_file', meta_info['gt_roof_csv_file']])
        csv_writer.writerow(['gt_footprint_csv_file', meta_info['gt_footprint_csv_file']])
        csv_writer.writerow(['vis_dir', meta_info['vis_dir']])
        csv_writer.writerow([''])
        for mask_type in ['roof', 'footprint']:
            csv_writer.writerow([mask_type])
            csv_writer.writerow([segmentation_eval_results[mask_type]])
            csv_writer.writerow(['F1 Score', segmentation_eval_results[mask_type]['F1_score']])
            csv_writer.writerow(['Precision', segmentation_eval_results[mask_type]['Precision']])
            csv_writer.writerow(['Recall', segmentation_eval_results[mask_type]['Recall']])
            csv_writer.writerow(['True Positive', segmentation_eval_results[mask_type]['TP']])
            csv_writer.writerow(['False Positive', segmentation_eval_results[mask_type]['FP']])
            csv_writer.writerow(['False Negative', segmentation_eval_results[mask_type]['FN']])
            csv_writer.writerow([''])
        csv_writer.writerow(['Length Error Classification'])
        csv_writer.writerow([str(interval) for interval in offset_eval_results['classify_interval']])
        csv_writer.writerow([str(error) for error in offset_eval_results['length_error_each_class']])
        csv_writer.writerow([str(mean_error) for mean_error in offset_eval_results['region_mean']])
        csv_writer.writerow([''])
        csv_writer.writerow(['Angle Error Classification'])
        csv_writer.writerow([str(error) for error in angle_eval_results['angle_error_each_class']])
        csv_writer.writerow([''])
        csv_writer.writerow(['Error Vector'])
        csv_writer.writerow(['aEPE', error_vector_results['aEPE']])
        csv_writer.writerow(['aAE', error_vector_results['aAE']])

        csv_writer.writerow([''])


ALL_MODELS = ['bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0'
            'bc_v005.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_5.0',
            'bc_v005.03_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_conv10',
            'bc_v005.04_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_weight_2.0', 
            'bc_v005.05_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1',
            'bc_v005.06_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10', 'bc_v005.06.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10', 'bc_v005.07_offset_rcnn_r50_2x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10', 
            'bc_v005.08_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar', 
            'bc_v005.08.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_direct', 
            'bc_v005.08.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin', 
            'bc_v005.08.03_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin_no_norm',
            'bc_v005.09_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offset_augmentation',
            'bc_v005.09.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin_no_norm_augmentation',
            'bc_v005.09.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin_no_norm_rotate_augmentation',
            'bc_v005.09.03_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin_no_norm_any_angle_rotate_augmentation',
            'bc_v005.09.04_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin_no_norm_rotate_augmentation_4_angles',
            'bc_v005.09.05_offset_rcnn_r50_2x_v1_5city_trainval_roof_mask_building_bbox_x_y_rotate_augmentation_4_angles',
            'bc_v005.09.06_offset_rcnn_r50_3x_v1_5city_trainval_roof_mask_building_bbox_x_y_rotate_augmentation_4_angles',
            'bc_v005.09.08_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_rotate_augmentation_4_angles_14_size',
            'bc_v005.09.09_offset_rcnn_r50_1x_v2_5city_trainval_roof_mask_building_bbox_rotate_augmentation_4_angles',
            'bc_v008.01_mask_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_urban3d',
            'bc_v008.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_urban3d',
            'bc_v008.03_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_urban3d_height',
            'bc_v008.04_mask_rcnn_r50_1x_v1_5city_trainval_roof_mask_roof_bbox_urban',
            'bc_v008.05_mask_rcnn_r50_1x_v1_5city_trainval_footprint_mask_footprint_bbox_urban3d',
            'bc_v008.06_mask_rcnn_r50_1x_v1_5city_trainval_footprint_mask_building_bbox_urban3d',
            'bc_v009_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_only_offset',
            'bc_v009.01_offset_rcnn_r50_2x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10_only_offset',
            'bc_v010.01_semi_supervised_offset_rcnn_r50_1x_v1',
            'bc_v010.02_semi_supervised_offset_rcnn_r50_1x_v1_lr_0.02',
            'bc_v010.03_semi_supervised_offset_rcnn_r50_1x_v1_without_footprint',
            'bc_v010.05.01_real_semi_supervised_offset_rcnn_r50_1x_v1_semi_urban3d_bc',
            'bc_v010.06.01_real_semi_supervised_offset_rcnn_r50_1x_v1_urban3d',
            'bc_v010.07.01_real_semi_supervised_offset_rcnn_r50_1x_v1_semi_urban3d_bc_bs2',
            'bc_v010.08.01_real_semi_supervised_offset_rcnn_r50_1x_v1_bc_beijing',
            'bc_v010.09.01_real_semi_supervised_offset_rcnn_r50_1x_v1_bc_beijing_pretrain',
            'bc_v010.10.01_real_semi_supervised_offset_rcnn_r50_1x_v1_semi_urban3d_bc_wo_building_bbox',
            'bc_v010.11.01_real_semi_supervised_offset_rcnn_r50_1x_v1_semi_urban3d_bc_bs2_pretrain',
            'bc_v010.12.01_real_semi_supervised_offset_rcnn_r50_1x_v1_bc_beijing_pretrain',
            'bc_v010.13.01_real_semi_supervised_offset_rcnn_r50_1x_v1_bc_wo_building_bbox',
            'bc_v010.14.01_real_semi_supervised_offset_rcnn_r50_1x_v1_bc_wo_building_bbox_angle',
            'bc_v010.15.01_real_semi_supervised_offset_rcnn_r50_1x_v1_semi_urban3d_bc_wo_building_bbox_rpn',
            'bc_v010.15.02_real_semi_supervised_offset_rcnn_r50_1x_v1_semi_urban3d_bc_wo_building_bbox_rpn_pretrain',
            'bc_v010.15.03_real_semi_supervised_offset_rcnn_r50_1x_v1_semi_urban3d_bc_wo_building_bbox_rpn_pretrain_lr0.04',
            'bc_v010.16.01_real_semi_supervised_offset_rcnn_r50_1x_v1_bc_wo_building_bbox_rpn',
            'bc_v010.16.02_real_semi_supervised_offset_rcnn_r50_1x_v1_bc_wo_building_bbox_rpn_pretrain',
            'bc_v010.16.03_real_semi_supervised_offset_rcnn_r50_1x_v1_bc_wo_building_bbox_rpn_pretrain_lr0.04',
            'bc_v010.17.01_real_semi_supervised_offset_rcnn_r50_1x_v1_bc_wo_building_bbox_rpn_semi_angle',
            'bc_v010.17.02_real_semi_supervised_offset_rcnn_r50_1x_v1_semi_urban3d_bc_wo_building_bbox_rpn_semi_angle',
            'bc_v011.01_offset_rcnn_r50_1x_v1_with_edge',
            'bc_v012.01.01_r50_1x_v1_offset_field',
            'bc_v012.02.01_r50_1x_v1_pixel_offset',
            'bc_v013.01.01_offset_rcnn_with_side_face_reweight',
            'bc_v014.01.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_coordconv',
            'bc_v015.01.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_non_local',
            ]

if __name__ == '__main__':
    # models = ['bc_v005.08.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin', 'bc_v005.08.03_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin_no_norm']
    # models = ['bc_v005.07_offset_rcnn_r50_2x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10']
    models = [model for model in ALL_MODELS[1:] if 'v005.09.09' in model]
    # models = ['bc_v006.05_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_angle']
    # models = ['bc_v006.01_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_linear_50_50']
    # cities = ['jinan', 'shanghai', 'beijing','chengdu', 'haerbin']
    # cities = ['dalian', 'xian', 'xian_fixed']
    cities = ['dalian', 'xian']
    # cities = ['xian']
    # cities = ['dalian', 'xian_fixed']
    # cities = ['urban3d']

    with_only_vis = False
    with_offset = True
    replace_pred_roof = False
    replace_pred_offset = False
    offset_model = 'footprint2roof'

    for model in models:
        version = model.split('_')[1]
        with_height = False
        with_only_offset = False
        if 'v006' in model or 'v008.03' in model or 'v010.14.01' in model:
            with_height = True
        if 'v009' in model:
            with_only_offset = True

        if 'v008.01' in model or 'v008.04' in model:
            with_offset = False

        for city in cities:
            print(f"========== {model} ========== {city} ==========")

            output_dir = f'./data/buildchange/statistic/{model}/{city}'
            bstool.mkdir_or_exist(output_dir)
            vis_boundary_dir = f'./data/buildchange/vis/{model}/{city}/boundary'
            bstool.mkdir_or_exist(vis_boundary_dir)
            if replace_pred_roof:
                vis_offset_dir = f'./data/buildchange/vis/{model}/{city}/offset_replace'
            else:
                vis_offset_dir = f'./data/buildchange/vis/{model}/{city}/offset'
            bstool.mkdir_or_exist(vis_offset_dir)
            if replace_pred_roof:
                summary_file = f'./data/buildchange/summary/{model}/{model}_{city}_eval_summary_replace_roof.csv'
            else:
                summary_file = f'./data/buildchange/summary/{model}/{model}_{city}_eval_summary.csv'
            if replace_pred_offset:
                summary_file = f'./data/buildchange/summary/{model}/{model}_{city}_eval_summary_replace_offset.csv'
            else:
                summary_file = f'./data/buildchange/summary/{model}/{model}_{city}_eval_summary.csv'
            bstool.mkdir_or_exist(f'./data/buildchange/summary/{model}')
            
            if city == 'xian':
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}_fine.json'
                gt_roof_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt.csv'
                gt_footprint_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_footprint_gt.csv'
                image_dir = f'./data/buildchange/v0/xian_fine/images'
            elif city == 'xian_fixed':
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_xian_fine.json'
                gt_roof_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt_fixed.csv'
                gt_footprint_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_footprint_gt_fixed.csv'
                image_dir = f'./data/buildchange/v0/xian_fine/images'
            elif city == 'dalian':
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}_fine.json'
                gt_roof_csv_file = f'./data/buildchange/v0/dalian_fine/dalian_fine_2048_roof_gt.csv'
                gt_footprint_csv_file = f'./data/buildchange/v0/dalian_fine/dalian_fine_2048_footprint_gt.csv'
                image_dir = f'./data/buildchange/v0/dalian_fine/images'
            elif city == 'urban3d':
                imageset = 'val'
                anno_file = f'./data/urban3d/v2/coco/annotations/urban3d_v2_val_JAX_OMA.json'
                gt_roof_csv_file = './data/urban3d/weijia/urban3d_jax_oma_val_roof_offset_gt_simple_subcsv_merge.csv'
                gt_footprint_csv_file = './data/urban3d/weijia/urban3d_jax_oma_val_orgfootprint_offset_gt_simple_subcsv_merge.csv'
                image_dir = f'./data/urban3d/v1/val/images'
                # offset_model = 'roof2footprint'
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
            
            roof_csv_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_{city}_roof_merged.csv'
            rootprint_csv_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_{city}_footprint_merged.csv'

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
                                        with_offset=with_offset,
                                        with_height=with_height,
                                        show=True,
                                        replace_pred_roof=replace_pred_roof,
                                        replace_pred_offset=replace_pred_offset,
                                        with_only_offset=with_only_offset,
                                        offset_model=offset_model)

            title = city + version
            if with_only_vis is False:
                # evaluation
                segmentation_eval_results = evaluation.segmentation()
                offset_eval_results = evaluation.offset_length_classification(title=title)
                angle_eval_results = evaluation.offset_angle_classification(title=title)
                error_vector_results = evaluation.offset_error_vector(title=title)
                if with_height:
                    evaluation.height(percent=100, title=title)

                meta_info = dict(summary_file=summary_file,
                                model=model,
                                anno_file=anno_file,
                                gt_roof_csv_file=gt_roof_csv_file,
                                gt_footprint_csv_file=gt_footprint_csv_file,
                                vis_dir=vis_boundary_dir)
                write_results2csv([segmentation_eval_results, offset_eval_results, angle_eval_results, error_vector_results], meta_info)

                # vis
                # evaluation.visualization_boundary(image_dir=image_dir, vis_dir=vis_boundary_dir)
                # for with_footprint in [True, False]:
                #     evaluation.visualization_offset(image_dir=image_dir, vis_dir=vis_offset_dir, with_footprint=with_footprint)

            else:
                # error_vector_results = evaluation.offset_error_vector(title=title)
                evaluation.visualization_boundary(image_dir=image_dir, vis_dir=vis_boundary_dir)
                for with_footprint in [True, False]:
                    evaluation.visualization_offset(image_dir=image_dir, vis_dir=vis_offset_dir, with_footprint=with_footprint)
