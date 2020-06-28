import os
import os.path as osp
import tempfile
import pandas
import csv
import itertools

import mmcv
import bstool
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from terminaltables import AsciiTable


class DetEval():
    def __init__(self, 
                 results,
                 ann_file,
                 json_prefix,
                 CLASSES=('building')):
        self.json_prefix = json_prefix
        if isinstance(results, str):
            self.results = mmcv.load(results)
        else:
            self.results = results
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=CLASSES)
        self.img_ids = self.coco.get_img_ids()

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self.img_ids)):
            img_id = self.img_ids[idx]
            info = self.coco.load_imgs([img_id])[0]
            image_name = info['file_name']
            ori_image_name = image_name.split("__")[1]
            bboxes, masks, scores = results[ori_image_name]

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
                rles = maskUtils.frPyObjects([masks[i]], 2048, 2048)
                rle = maskUtils.merge(rles)
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode()
                data['segmentation'] = rle
                segm_json_results.append(data)

        return bbox_json_results, segm_json_results

    def results2json(self):
        result_files = dict()
        json_results = self._segm2json(self.results)

        result_files['bbox'] = f'{self.json_prefix}.bbox.json'
        result_files['segm'] = f'{self.json_prefix}.segm.json'
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])

        return result_files

    def evaluation(self, 
                   metric=['bbox', 'segm'],
                   classwise=False,
                   proposal_nums=(100, 300, 1000),
                   iou_thrs=np.arange(0.5, 0.96, 0.05)):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm']
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
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids

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
