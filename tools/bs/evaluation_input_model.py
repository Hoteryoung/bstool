import argparse

import bstool


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument(
        '--model',
        type=str,
        default='demo')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.model == 'demo':
        raise(RuntimeError("Please input the valid model name"))
    else:
        models = [args.model]
    
    cities = ['xian', 'dalian']

    for model in models:
        version = model.split('_')[1]
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

            title = city + "_" + version

            evaluation.segmentation()
            evaluation.offset_length_classification(title=title)
            evaluation.offset_angle_classification(title=title)
            evaluation.offset_error_vector(title=title)
            if city == 'xian':
                continue
            evaluation.height(percent=100, title=title)