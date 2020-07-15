import bstool


if __name__ == '__main__':
    # model = 'bc_v006.01_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox_linear_50_50'
    model = 'bc_v006_height_rcnn_r50_1x_v1_5city_trainval_roof_mask_building_bbox'
    # cities = ['jinan', 'shanghai', 'beijing','chengdu', 'haerbin']
    cities = ['dalian_fine']
    
    for city in cities:
        print(f"Start processing {city}")

        output_dir = './data/buildchange/v0/statistic'

        if 'xian' in city or 'dalian' in city:
            imageset = 'val'
        else:
            imageset = 'train'

        anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_{imageset}_{city}.json'
        pkl_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_{city}_coco_results.pkl'
        
        gt_roof_csv_file = f'./data/buildchange/v0/{city}/{city}_2048_roof_gt.csv'
        gt_footprint_csv_file = f'./data/buildchange/v0/{city}/{city}_2048_footprint_gt.csv'
        
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
                                       with_height=True,
                                       show=True)

        evaluation.height(percent=100, title=city)
        print(f"Finish processing {city}")