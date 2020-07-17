import os
import os.path as osp
import tempfile
import pandas
import csv
import itertools
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
from terminaltables import AsciiTable

import mmcv
import bstool


class COCOEvalExtend():
    def __init__(self,
                 ann_file,
                 png_dir=None,
                 png_format='.png',
                 csv_file=None,
                 json_prefix=None,
                 image_size=(2048, 2048),
                 subclass=255):
        self.png_dir = png_dir
        self.png_format = png_format
        self.csv_file = csv_file
        self.json_prefix = json_prefix
        self.image_size = image_size
        self.subclass = subclass

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.img_ids = self.coco.get_img_ids()

        if png_dir is not None and csv_file is None:
            self.json_results = self._png2json(png_dir)
        elif csv_file is not None and png_dir is None:
            self._csv2json(csv_file)
        else:
            raise(RuntimeError(f"do not support png_dir: {png_dir} and csv_file: {csv_file}"))

    def dump_json_results(self):
        result_files = dict()

        result_files['bbox'] = f'{self.json_prefix}.bbox.json'
        result_files['segm'] = f'{self.json_prefix}.segm.json'
        mmcv.dump(self.json_results[0], result_files['bbox'])
        mmcv.dump(self.json_results[1], result_files['segm'])

        return result_files

    def _png2json(self, png_dir):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self.img_ids)):
            img_id = self.img_ids[idx]
            info = self.coco.load_imgs([img_id])[0]
            image_name = bstool.get_basename(info['file_name'])

            png_file = os.path.join(png_dir, image_name + self.png_format)
            objects = bstool.mask_parse(png_file, subclasses=[self.subclass, self.subclass], clean_polygon_flag=True, with_opencv=True)
            
            masks = [obj['mask'] for obj in objects]
            bboxes = [bstool.mask2bbox(mask) for mask in masks]

            for bbox, mask in zip(bboxes, masks):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = bstool.xyxy2xywh(bbox)
                data['score'] = 1.0
                data['category_id'] = 1

                rles = maskUtils.frPyObjects([mask], self.image_size[0], self.image_size[1])
                rle = maskUtils.merge(rles)
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode()
                data['segmentation'] = rle

                bbox_json_results.append(data)
                segm_json_results.append(data)

        return bbox_json_results, segm_json_results

    def _csv2json(self, csv_file):
        pass

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

        result_files = self.dump_json_results()

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


if __name__ == '__main__':
    # ann_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_val_xian_fine_origin.json'
    # # png_dir = '/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/weijia'
    # png_dir = '/data/buildchange/v0/xian_fine/png_gt'
    # json_prefix = '/home/jwwangchn/Documents/100-Work/170-Codes/mmdetv2-bc/results/pred'

    ann_file = '/mnt/lustrenew/liweijia/data/crowdAI/val/annotation-small.json'
    png_dir = '/mnt/lustrenew/liweijia/data/crowdAI/val/val-annotation-small-png-01'
    json_prefix = '/mnt/lustre/wangjinwang/buildchange/codes/bstool/data/buildchange/v0/crowdAI/crowdAI'

    bstool.mkdir_or_exist('/mnt/lustre/wangjinwang/buildchange/codes/bstool/data/buildchange/v0/crowdAI')
    
    coco_eval_extend = COCOEvalExtend(ann_file=ann_file,
                                      png_dir=png_dir,
                                      csv_file=None,
                                      json_prefix=json_prefix,
                                      image_size=(300, 300))
    
    coco_eval_extend.evaluation(metric=['bbox', 'segm'],
                                classwise=False,
                                proposal_nums=(100, 300, 1000),
                                iou_thrs=np.arange(0.5, 0.96, 0.05))