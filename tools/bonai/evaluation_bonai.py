# -*- encoding: utf-8 -*-
'''
@File    :   evaluation_public.py
@Time    :   2020/12/30 22:29:35
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   公开数据集的评估代码，较为重要
'''


import bstool
import csv
import argparse


def write_results2csv(results, meta_info=None):
    """Write the evaluation results to csv file

    Args:
        results (list): list of result
        meta_info (dict, optional): The meta info about the evaluation (file path of ground truth etc.). Defaults to None.
    """
    print("meta_info: ", meta_info)
    segmentation_eval_results = results[0]
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

        csv_writer.writerow([''])

# Save the FULL CONFIG NAME of models which need to be evaluated, you can use the short name (bc_vXXX.XX.XX) to filter the model
ALL_MODELS = [
            'bonai_v001.01.01_loft_foa',
            'bonai_v001.01.02_loft_foa_single_gpu',
            'bonai_v001.02.01_loft_foa_r18',
            'bonai_v001.02.02_mask_rcnn_r18'
            ]

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval on semantic segmentation')
    parser.add_argument(
        '--version',
        type=str,
        default='bc_v100.01.09', 
        help='model name (version) for evaluation')
    parser.add_argument(
        '--model',
        type=str,
        default='', 
        help='full model name for evaluation')
    parser.add_argument(
        '--city',
        type=str,
        default='', 
        help='dataset city for evaluation')

    args = parser.parse_args()

    return args

def get_model_shortname(model_name):
    return "bonai" + "_" + model_name.split('_')[1]

class EvaluationParameters:
    def __init__(self, city, model):
        # flags
        self.with_vis = False
        self.with_only_vis = False
        self.with_only_pred = True
        self.with_image = True
        self.with_offset = True
        self.save_merged_csv = False

        # baseline models
        self.baseline_models = ['bonai_v001.02.02']

        # basic info
        self.city = city
        self.model = model
        self.score_threshold = 0.4
        self.dataset_root = "../mmdetv2-bc/data/BONAI"
        self.csv_groundtruth_root = "../mmdetv2-bc/data/BONAI/csv"
        self.pred_result_root = "../mmdetv2-bc/results/bonai"

        # dataset file
        self.anno_file = f'{self.dataset_root}/coco/bonai_shanghai_xian_test.json'
        self.test_image_dir = f'{self.dataset_root}/test/images'

        # csv ground truth files
        self.gt_roof_csv_file = f'{self.csv_groundtruth_root}/shanghai_xian_v3_merge_val_roof_crop1024_gt_minarea500.csv'
        self.gt_footprint_csv_file = f'{self.csv_groundtruth_root}/shanghai_xian_v3_merge_val_footprint_crop1024_gt_minarea500.csv'

        # detection result files
        self.mmdetection_pkl_file = f'{self.pred_result_root}/{model}/{model}_{city}_coco_results.pkl'
        self.csv_info = 'merged' if self.save_merged_csv else 'splitted'
        self.pred_roof_csv_file = f'{self.pred_result_root}/{model}/{model}_roof_{self.csv_info}.csv'
        self.pred_rootprint_csv_file = f'{self.pred_result_root}/{model}/{model}_footprint_{self.csv_info}.csv'

        # vis
        self.vis_boundary_dir = f'{self.dataset_root}/vis/{model}/boundary' + ("_pred" if self.with_only_pred else "")
        self.vis_offset_dir = f'{self.dataset_root}/vis/{model}/offset'

        # summary
        self.summary_file = f'{self.dataset_root}/summary/{model}/{model}_eval_summary_{self.csv_info}.csv'

    def post_processing(self):
        bstool.mkdir_or_exist(self.vis_boundary_dir)
        bstool.mkdir_or_exist(self.vis_offset_dir)
        bstool.mkdir_or_exist(f'{self.dataset_root}/summary/{model}')
        
if __name__ == '__main__':
    args = parse_args()

    eval_parameters = EvaluationParameters(city = args.city, model = args.model)
    # baseline (mask rcnn) or LOFT
    eval_parameters.with_offset =  False if get_model_shortname(args.model) in eval_parameters.baseline_models else True
    
    print(f"========== {args.model} ========== {args.city} ==========")

    # not used
    output_dir = f'./data/buildchange/statistic/{args.model}/{args.city}'
    bstool.mkdir_or_exist(output_dir)

    evaluation = bstool.Evaluation(model = eval_parameters.model,
                                    anno_file = eval_parameters.anno_file,
                                    pkl_file = eval_parameters.mmdetection_pkl_file,
                                    gt_roof_csv_file = eval_parameters.gt_roof_csv_file,
                                    gt_footprint_csv_file = eval_parameters.gt_footprint_csv_file,
                                    roof_csv_file = eval_parameters.pred_roof_csv_file,
                                    rootprint_csv_file = eval_parameters.pred_rootprint_csv_file,
                                    iou_threshold = 0.1,
                                    score_threshold = eval_parameters.score_threshold,
                                    output_dir = output_dir,
                                    with_offset = eval_parameters.with_offset,
                                    show = False,
                                    save_merged_csv = eval_parameters.save_merged_csv)

    if eval_parameters.with_only_vis is False:
        # evaluation
        if evaluation.dump_result:
            # calculate the F1 score
            segmentation_eval_results = evaluation.segmentation()
            epe_results = evaluation.offset_error_vector()
            print("Offset EPE: ", epe_results)
            meta_info = dict(summary_file = eval_parameters.summary_file,
                            model = eval_parameters.model,
                            anno_file = eval_parameters.anno_file,
                            gt_roof_csv_file = eval_parameters.gt_roof_csv_file,
                            gt_footprint_csv_file = eval_parameters.gt_footprint_csv_file,
                            vis_dir = eval_parameters.vis_boundary_dir)
            write_results2csv([segmentation_eval_results], meta_info)
            result_dict = {"Roof F1: ": segmentation_eval_results['roof']['F1_score'],
                           "Roof Precition: ": segmentation_eval_results['roof']['Precision'],
                           "Roof Recall: ": segmentation_eval_results['roof']['Recall'],
                           "Footprint F1: ": segmentation_eval_results['footprint']['F1_score'],
                           "Footprint Precition: ": segmentation_eval_results['footprint']['Precision'],
                           "Footprint Recall: ": segmentation_eval_results['footprint']['Recall']}
            print("result_dict: ", result_dict)
        else:
            print('!!!!!!!!!!!!!!!!!!!!!! ALl the results of images are empty !!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # vis
        if eval_parameters.with_vis:
            # generate the vis results
            evaluation.visualization_boundary(image_dir = eval_parameters.test_image_dir, 
                                              vis_dir = eval_parameters.vis_boundary_dir, 
                                              with_gt = True)
            # draw offset in the image (not used in this file)
            # for with_footprint in [True, False]:
            #     evaluation.visualization_offset(image_dir=image_dir, vis_dir=vis_offset_dir, with_footprint=with_footprint)
    else:
        # generate the vis results
        evaluation.visualization_boundary(image_dir = eval_parameters.test_image_dir, 
                                          vis_dir = eval_parameters.vis_boundary_dir, 
                                          with_gt = True, 
                                          with_only_pred = eval_parameters.with_only_pred, 
                                          with_image = eval_parameters.with_image)
        # draw offset in the image (not used in this file)
        # for with_footprint in [True, False]:
        #     evaluation.visualization_offset(image_dir=image_dir, vis_dir=vis_offset_dir, with_footprint=with_footprint)
