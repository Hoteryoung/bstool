import bstool
import csv

def write_results2csv(results, meta_info=None):
    print("meta_info: ", meta_info)
    segmentation_eval_results, offset_eval_results, angle_eval_results, error_vector_results = results
    with open(meta_info['summary_file'], 'w') as summary:
        csv_writer = csv.writer(summary, delimiter=',')
        csv_writer.writerow(['Meta Info'])
        csv_writer.writerow(['model', model])
        csv_writer.writerow(['anno_file', anno_file])
        csv_writer.writerow(['gt_roof_csv_file', gt_roof_csv_file])
        csv_writer.writerow(['gt_footprint_csv_file', gt_footprint_csv_file])
        csv_writer.writerow(['vis_dir', vis_dir])
        csv_writer.writerow([''])
        for mask_type in ['roof', 'footprint']:
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


ALL_MODELS = ['bc_v005.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0',
            'bc_v005.01.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_2.0_test_nms',
            'bc_v005.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_offsetweight_5.0',
            'bc_v005.03_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_conv10',
            'bc_v005.04_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_weight_2.0', 
            'bc_v005.05_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1',
            'bc_v005.06_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10', 'bc_v005.06.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10_no_ignore', 'bc_v005.06.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10', 'bc_v005.07_offset_rcnn_r50_2x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10', 
            'bc_v005.08_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar', 
            'bc_v005.08.01_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_direct', 
            'bc_v005.08.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin', 
            'bc_v005.08.03_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin_no_norm']

if __name__ == '__main__':
    # models = ['bc_v005.08.02_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin', 'bc_v005.08.03_offset_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_polar_cos_sin_no_norm']
    # models = ['bc_v005.07_offset_rcnn_r50_2x_v1_5city_trainval_roof_mask_building_bbox_smooth_l1_offsetweight_2.0_conv10']
    models = [model for model in ALL_MODELS if 'v005' in model]
    # models = ['bc_v006.05_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_angle']
    # models = ['bc_v006.01_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_linear_50_50']
    # cities = ['jinan', 'shanghai', 'beijing','chengdu', 'haerbin']
    # cities = ['jinan', 'shanghai', 'beijing','chengdu', 'haerbin']
    cities = ['dalian', 'xian', 'xian_fixed']
    cities = ['dalian', 'xian', 'urban3d']

    with_only_vis = False

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
            summary_file = f'./data/buildchange/summary/{model}/{model}_{city}_eval_summary.csv'
            bstool.mkdir_or_exist(f'./data/buildchange/summary/{model}')
            
            if city == 'xian':
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}_fine.json'
                gt_roof_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt.csv'
                gt_footprint_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_footprint_gt.csv'
                image_dir = f'./data/buildchange/v0/{city}_fine/images'
            elif city == 'xian_fixed':
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_xian_fine.json'
                gt_roof_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_roof_gt_fixed.csv'
                gt_footprint_csv_file = './data/buildchange/v0/xian_fine/xian_fine_2048_footprint_gt_fixed.csv'
                image_dir = f'./data/buildchange/v0/{city}_fine/images'
            elif city == 'dalian':
                imageset = 'val'
                anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}_fine.json'
                gt_roof_csv_file = f'./data/buildchange/v0/dalian_fine/dalian_fine_2048_roof_gt.csv'
                gt_footprint_csv_file = f'./data/buildchange/v0/dalian_fine/dalian_fine_2048_footprint_gt.csv'
                image_dir = f'./data/buildchange/v0/{city}_fine/images'
            elif city == 'urban3d':
                imageset = 'val'
                anno_file = f'./data/urban3d/v2/coco/annotations/urban3d_v2_val_JAX_OMA.json'
                gt_roof_csv_file = './data/urban3d/v0/val/urban3d_2048_JAX_OMA_roof_gt.csv'
                gt_footprint_csv_file = './data/urban3d/v0/val/urban3d_2048_JAX_OMA_footprint_gt.csv'
                image_dir = f'./data/urban3d/v1/val/images'
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
                segmentation_eval_results = evaluation.segmentation()
                offset_eval_results = evaluation.offset_length_classification(title=title)
                angle_eval_results = evaluation.offset_angle_classification(title=title)
                error_vector_results = evaluation.offset_error_vector(title=title)
                if with_height:
                    evaluation.height(percent=100, title=title)
                evaluation.visualization(image_dir=image_dir, vis_dir=vis_dir)

                meta_info = dict(summary_file=summary_file,
                                model=model,
                                anno_file=anno_file,
                                gt_roof_csv_file=gt_roof_csv_file,
                                gt_footprint_csv_file=gt_footprint_csv_file,
                                vis_dir=vis_dir)
                write_results2csv([segmentation_eval_results, offset_eval_results, angle_eval_results, error_vector_results], meta_info)

            else:
                evaluation.visualization(image_dir=image_dir, vis_dir=vis_dir)