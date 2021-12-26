# -*- encoding: utf-8 -*-
'''
@File    :   evaluation_public_ap50.py
@Time    :   2020/12/30 22:29:23
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   计算模型的 AP50 指标
'''


import bstool
import csv
import argparse
from boundary_iou.coco_instance_api.coco import COCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from terminaltables import AsciiTable
import numpy as np
import mmcv
import pandas

class CSV2Json():
    def __init__(self, ann_file, csv_file, json_prefix):
        self.json_prefix = json_prefix
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=('building',))
        self.img_ids = self.coco.getImgIds()

        self.results = self._parse_results(csv_file)

        self.results2json()

    def results2json(self):
        result_files = dict()
        json_results = self._segm2json(self.results)

        result_files['bbox'] = f'{self.json_prefix}.bbox.json'
        result_files['segm'] = f'{self.json_prefix}.segm.json'
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])

        return result_files

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self.img_ids)):
            img_id = self.img_ids[idx]
            info = self.coco.loadImgs([img_id])[0]
            image_name = bstool.get_basename(info['file_name'])
            bboxes, masks, scores = results[image_name]

            for i in range(len(bboxes)):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = bstool.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = 1
                bbox_json_results.append(data)
            
            for i in range(len(masks)):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = bstool.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = 1
                rles = maskUtils.frPyObjects([masks[i]], 1024, 1024)
                rle = maskUtils.merge(rles)
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode()
                data['segmentation'] = rle
                segm_json_results.append(data)

        return bbox_json_results, segm_json_results

    def _parse_results(self, csv_file):
        results = dict()
        csv_parser = bstool.CSVParse(csv_file)
        objects = csv_parser.objects
        image_name_list = csv_parser.image_name_list

        for ori_image_name in image_name_list:
            buildings = dict()
            buildings = objects[ori_image_name]
            masks = [bstool.polygon2mask(building['polygon']) for building in buildings]
            bboxes = [bstool.mask2bbox(mask) for mask in masks]
            scores = [building['score'] for building in buildings]

            results[ori_image_name] = [bboxes, masks, scores]

        return results

    def evaluation(self, 
                   metric=['bbox', 'segm'],
                   classwise=False,
                   proposal_nums=(100, 300, 1000),
                   iou_thrs=np.arange(0.5, 0.96, 0.05)):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'boundary']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        result_files = self.results2json()

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            print(msg)
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print('The testing results of the whole dataset is empty.')
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iouType="boundary")
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids

            cocoEval.params.maxDets = [2, 25, 250]

            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)

            metric_items = [
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
            ]
            for i in range(len(metric_items)):
                key = f'{metric}_{metric_items[i]}'
                val = float(f'{cocoEval.stats[i]:.3f}')
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')
        
        return eval_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval on semantic segmentation')
    parser.add_argument(
        '--version',
        type=str,
        default='bc_v100.02.08', 
        help='dataset for evaluation')
    parser.add_argument(
        '--model',
        type=str,
        default='', 
        help='full model name for evaluation')
    parser.add_argument(
        '--city',
        type=str,
        default='shanghai_xian_public', 
        help='dataset for evaluation')

    args = parser.parse_args()

    return args

class EvaluationParameters:
    def __init__(self, city, model):
        # flags
        self.save_merged_csv = False

        # basic info
        self.city = city
        self.model = model
        self.dataset_root = "../mmdetv2-bc/data/BONAI"
        self.csv_groundtruth_root = "../mmdetv2-bc/data/BONAI/csv"
        self.pred_result_root = "../mmdetv2-bc/results/bonai"

        # dataset file
        self.anno_file = f'{self.dataset_root}/coco/bonai_shanghai_xian_test.json'
        self.test_image_dir = f'{self.dataset_root}/test/images'

        # detection result files
        self.csv_info = 'merged' if self.save_merged_csv else 'splitted'
        self.pred_footprint_csv_file = f'{self.pred_result_root}/{model}/{model}_footprint_{self.csv_info}.csv'
        self.footprint_json_prefix = f'{self.pred_result_root}/{model}/{model}_{city}_footprint_{self.csv_info}'

        # summary
        self.summary_file = f'{self.dataset_root}/summary/{model}/{model}_eval_summary_{self.csv_info}_boundary_ap50.csv'

if __name__ == '__main__':
    args = parse_args()

    predefined_models = ['bonai_v001.01.01_loft_foa',
                        'bonai_v001.01.02_loft_foa_single_gpu',
                        'bonai_v001.01.03_loft_foa_from_scratch',
                        'bonai_v001.01.04_loft_foa_single_gpu_warmup_150',
                        'bonai_v001.01.05_loft_foa_single_gpu_warmup_600',
                        'bonai_v001.02.01_loft_foa_r18',
                        'bonai_v001.02.02_mask_rcnn_r18',
                        'bonai_v001.03.01_mask_rcnn_single_gpu',
                        'bonai_v001.03.02_mask_rcnn_from_scratch',
                        'bonai_v001.03.03_mask_rcnn_four_gpus']

    predefined_models = predefined_models if args.model is "" else [args.model]

    for model in predefined_models:
        eval_parameters = EvaluationParameters(city = args.city, model = model)

        print(f"\n========== {eval_parameters.model} ========== {eval_parameters.city} ==========")
        
        csv2json_core = CSV2Json(ann_file=eval_parameters.anno_file, 
                                csv_file=eval_parameters.pred_footprint_csv_file, 
                                json_prefix=eval_parameters.footprint_json_prefix)
        
        eval_results = csv2json_core.evaluation()

        print(eval_results, type(eval_results))
        pandas.DataFrame([eval_results]).to_csv(eval_parameters.summary_file)