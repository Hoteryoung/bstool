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


ALL_MODELS = ALL_MODELS = [
            'bc_v100.01.01_offset_rcnn_r50_1x_public_20201027_baseline',
            'bc_v100.01.02_offset_rcnn_r50_1x_public_20201027_lr0.01',
            'bc_v100.01.03_offset_rcnn_r50_1x_public_20201028_lr_0.02',
            'bc_v100.01.04_offset_rcnn_r50_2x_public_20201028_lr_0.02',
            'bc_v100.01.05_offset_rcnn_r50_2x_public_20201028_sample_num',
            'bc_v100.01.06_offset_rcnn_r50_3x_public_20201028_lr_0.02',
            'bc_v100.01.07_offset_rcnn_r50_2x_public_20201027_lr_0.02',
            'bc_v100.01.08_offset_rcnn_r50_2x_public_20201028_footprint_bbox_footprint_mask_baseline',
            'bc_v100.01.09_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline',
            'bc_v100.01.10_offset_rcnn_r50_2x_public_20201028_footprint_bbox_footprint_mask_baseline_no_aug',
            'bc_v100.01.11_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_no_aug',
            'bc_v100.01.12_offset_rcnn_r50_1x_public_20201028_footprint_bbox_footprint_mask_baseline_simple',
            'bc_v100.01.13_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_cascade_mask_rcnn',
            'bc_v100.01.14_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_pafpn',
            'bc_v100.01.15_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline',
            'bc_v100.01.16_offset_rcnn_r50_2x_public_20201028_lr_0.02_without_rotation_flip',
            'bc_v100.01.17_offset_rcnn_r50_2x_public_20201028_lr_0.02_without_rotation',
            'bc_v100.01.18_offset_rcnn_r50_2x_public_20201028_lr_0.02_only_arg',
            'bc_v100.01.19_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_roof',
            'bc_v100.01.20_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_cascade_mask_rcnn_roof',
            'bc_v100.01.21_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_pafpn_roof',
            'bc_v100.01.22_offset_rcnn_r50_2x_public_20201028_lr_0.02_without_image_rotation',
            'bc_v100.01.23_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_fix_parameter',
            'bc_v100.01.24_offset_rcnn_r50_2x_public_20201028_lr_0.02_48_epoch',
            'bc_v100.01.25_offset_rcnn_r50_2x_public_20201028_lr_0.02_without_rotation_48_epoch',
            'bc_v100.01.26_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_12epoch',
            'bc_v100.01.27_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_cascade_mask_rcnn_12epoch',
            'bc_v100.01.28_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_pafpn_12epoch',
            'bc_v100.01.29_offset_rcnn_r50_2x_public_20201028_lr_0.02_96epoch',
            'bc_v100.01.30_offset_rcnn_r50_2x_public_20201028_lr_0.02_without_image_rotation_1x',
            'bc_v100.01.31_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_12epoch_no_rotation',
            'bc_v100.01.32_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_cascade_mask_rcnn_12epoch_no_rotation',
            'bc_v100.01.33_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_pafpn_12epoch_no_rotation',
            'bc_v100.01.34_mask_rcnn_r50_1x_public_20201028_12epoch_no_rotation_0.01',
            'bc_v100.01.35_mask_rcnn_r50_1x_public_20201028_12epoch_no_rotation_0.01_nms',
            'bc_v100.01.36_cascade_rcnn_r50_1x_public_20201028_12epoch_no_rotation_0.01_nms',
            'bc_v100.01.37_panet_r50_1x_public_20201028_12epoch_no_rotation_0.01_nms',
            'bc_v100.01.38_mask_rcnn_r50_1x_public_20201028_12epoch_no_rotation_0.01_nms',
            'bc_v100.01.39_offset_rcnn_r50_1x_public_20201028_lr_0.02_without_image_rotation_1x',
            'bc_v100.01.40_offset_rcnn_r50_1x_public_20201028_lr_0.04_panet',
            'bc_v100.01.41_offset_rcnn_r50_1x_public_20201028_lr_0.04_cascade_mask_rcnn',
            'bc_v100.02.01_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles',
            'bc_v100.02.02_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_decouple',
            'bc_v100.02.03_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_minarea_500',
            'bc_v100.02.04_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_arg',
            'bc_v100.02.05_offset_rcnn_r50_2x_public_20201028_rotate_offset_2_angles',
            'bc_v100.02.06_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_flip',
            'bc_v100.02.07_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_without_image_rotation_flip',
            'bc_v100.02.08_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_without_image_rotation',
            'bc_v100.02.09_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_without_image_flip',
            'bc_v100.02.10_offset_rcnn_r50_1x_public_20201028_rotate_offset_4_angles_12epoch',
            'bc_v100.02.11_offset_rcnn_r50_2x_public_20201028_rotate_offset_2_angles_without_rotation',
            'bc_v100.02.12_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_only_share_conv',
            'bc_v100.02.13_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_share_conv_and_fc',
            'bc_v100.02.14_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_no_share_conv_fc',
            'bc_v100.02.16_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_48_epoch',
            'bc_v100.02.17_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_without_image_rotation',
            'bc_v100.03.01_semi_offset_rcnn_r50_2x_public_20201028_arg_pretrain',
            'bc_v100.03.02_semi_offset_rcnn_r50_2x_public_20201028_real_semi',
            'bc_v100.03.03_semi_offset_rcnn_r50_2x_public_20201028_real_semi_resume',
            'bc_v100.03.04_semi_offset_rcnn_r50_2x_public_20201028_arg_pretrain_lr0.01',
            'bc_v100.03.05_semi_offset_rcnn_r50_2x_public_20201028_real_semi_lr0.02',
            'bc_v100.03.06_semi_offset_rcnn_r50_2x_public_20201028_arg_pretrain_4x',
            'bc_v100.03.07_semi_offset_rcnn_r50_2x_public_20201028_real_semi_lr0.02_arg_google',
            'bc_v100.03.09_semi_offset_rcnn_r101_2x_public_20201028_arg_pretrain_resnet101',
            'bc_v100.03.10_offset_rcnn_r50_2x_public_20201028_footprint_bbox_footprint_mask_baseline_arg',
            'bc_v100.03.11_semi_offset_rcnn_r50_2x_public_20201028_arg_pretrain_no_footprint',
            'bc_v100.03.12_semi_offset_rcnn_r50_2x_public_20201028_real_semi_lr0.02_finetune_03.11',
            'bc_v100.03.13_semi_offset_rcnn_r50_2x_public_20201028_full_data',
            'bc_v100.03.14_semi_offset_rcnn_r50_2x_public_20201028_full_data_no_footprint',
            'bc_v100.03.15_semi_offset_rcnn_r50_2x_public_20201028_full_data_iou_loss',
            'bc_v100.03.16_semi_offset_rcnn_r50_2x_public_20201028_full_data_no_update_footprint_mask',
            'bc_v100.03.17_semi_offset_rcnn_r50_2x_public_20201028_full_data_fix_mask_bug',
            'bc_v100.03.18_semi_offset_rcnn_r50_2x_public_20201028_full_data_no_update_footprint_bbox',
            'bc_v100.03.19_semi_offset_rcnn_r50_2x_public_20201028_full_data_use_the_label_pos_smooth_l1',
            'bc_v100.03.20_semi_offset_rcnn_r50_2x_public_20201028_full_data_rewrite_mask_branch',
            'bc_v100.03.21_semi_offset_rcnn_r50_2x_public_20201028_full_data_rewrite_mask_branch_iou_loss',
            'bc_v100.03.22_semi_offset_rcnn_r50_2x_public_20201028_full_data_rewrite_mask_branch_iou_loss_weight_0.2',
            'bc_v100.03.23_semi_offset_rcnn_r50_2x_public_20201028_full_data_finetune_03.11',
            'bc_v100.03.24_semi_offset_rcnn_r50_2x_public_20201028_full_data_rewrite_mask_branch_iou_loss_weight_0.1',
            'bc_v100.03.25_semi_offset_rcnn_r50_2x_public_20201028_full_data_rewrite_mask_branch_iou_loss_weight_0.1_finetune_03.11',
            'bc_v100.03.26_semi_offset_rcnn_r50_2x_public_20201028_full_data_roof2footprint_finetune_03.11',
            'bc_v100.03.27_semi_offset_rcnn_r50_2x_public_20201028_roof2footprint_finetune_03.11_without_bbox',
            'bc_v100.03.28_semi_offset_rcnn_r50_2x_public_20201028_full_data_roof2footprint_finetune_03.11_lr0.01',
            'bc_v100.03.29_semi_offset_rcnn_r50_2x_public_20201028_roof2footprint_finetune_03.11_without_bbox_lr_0.01',
            'bc_v100.03.30_semi_offset_rcnn_r50_2x_public_20201028_full_data_loss_weight_1.0',
            'bc_v100.03.31_semi_offset_rcnn_r50_2x_public_20201028_full_data_roof2footprint_finetune_03.11_no_update_footprint_bbox',
            'bc_v100.03.32_semi_offset_rcnn_r50_2x_public_20201028_roof2footprint_finetune_03.11_without_bbox_experiment',
            'bc_v100.03.33_semi_offset_rcnn_r50_2x_public_20201028_roof2footprint_finetune_03.11_without_bbox_experiment',
            'bc_v100.03.34_semi_offset_rcnn_r50_2x_public_20201028_arg_roof2footprint',
            'bc_v100.03.36_semi_offset_rcnn_r50_2x_public_20201028_arg_google_debug',
            'bc_v100.03.35_semi_offset_rcnn_r50_2x_public_20201028_arg_pretrain',
            'bc_v100.03.37_semi_offset_rcnn_r50_2x_public_20201028_roof2footprintwithout_bbox_no_pretrain',
            'bc_v100.03.38_semi_offset_rcnn_r50_2x_public_20201028_arg_pretrain_without_rotation',
            'bc_v100.03.40_semi_offset_rcnn_r50_2x_public_20201028_arg_pretrain_without_rotation_with_foa',
            'bc_v100.03.39_semi_offset_rcnn_r50_2x_public_20201028_finetune_03.38_without_bbox_without_rotation',
            'bc_v100.03.41_semi_offset_rcnn_r50_2x_public_20201028_arg_pretrain_without_rotation_without_footprint_bbox',
            'bc_v100.03.42_semi_offset_rcnn_r50_2x_public_20201028_finetune_03.38_without_bbox_without_rotation_arg_google',
            'bc_v100.03.43_semi_offset_rcnn_r50_2x_public_20201028_finetune_03.38_without_bbox_without_rotation_arg_ms',
            'bc_v100.03.44_semi_offset_rcnn_r50_2x_public_20201028_finetune_03.38_without_bbox_without_rotation_arg_ms_google',
            'bc_v100.03.45_semi_offset_rcnn_r50_2x_public_20201028_without_bbox_without_rotation_arg_ms_google',
            'bc_v100.03.46_semi_offset_rcnn_r50_2x_public_20201028_arg_ms_google',
            'bc_v100.03.47_semi_offset_rcnn_r50_2x_public_20201028_finetune_03.38_without_bbox_without_rotation_arg_google_foa',
            'bc_v100.03.48_semi_offset_rcnn_r50_2x_public_20201028_finetune_03.38_without_bbox_without_rotation_arg_ms_foa',
            'bc_v100.03.49_semi_offset_rcnn_r50_2x_public_20201028_arg_ms_google_foa',
            'bc_v100.03.50_semi_offset_rcnn_r50_2x_public_20201028_arg_ms_google_without_rotation',
            'bc_v100.03.51_semi_offset_rcnn_r50_2x_public_20201028_finetune_03.38_without_bbox_without_rotation_arg_ms_google_foa',
            'bc_v100.03.52_semi_offset_rcnn_r50_2x_public_20201028_finetune_03.38_without_bbox_without_rotation_arg_full_ms_google_only',
            'bc_v100.03.53_semi_offset_rcnn_r50_2x_public_20201028_finetune_03.38_without_bbox_without_rotation_arg_full_ms_google_only_foa',
            'bc_v100.03.54_semi_offset_rcnn_r50_2x_public_20201028_arg_ms_google_without_rotation_lw_16.0',
            'bc_v100.03.55_semi_offset_rcnn_r50_2x_public_20201028_arg_ms_google_without_rotation_retraining',
            'bc_v100.03.56_semi_offset_rcnn_r50_2x_public_20201028_arg_ms_google_without_rotation_lw_4.0',
            'bc_v100.04.01_mask_rcnn_r50_1x_public_20201028_12epoch_no_rotation_0.01',
            'bc_v100.04.02_panet_rcnn_r50_1x_public_20201028_12epoch_no_rotation_0.01',
            'bc_v100.04.03_cascade_mask_rcnn_r50_1x_public_20201028_12epoch_no_rotation_0.01',
            'bc_v100.05.01_mask_rcnn_r50_pami_2x_lr0.01',
            'bc_v200.02.01_foa_1x_L1Loss_lr0.04_w32',
            'bc_v200.02.02_foa_1x_L1Loss_lr0.04_w64',
            'bc_v200.03.01_foa_1x_lr0.04_feature_size_5',
            'bc_v200.04.02_foa_1x_lr0.04_angle_0_180',
            'bc_v200.04.03_foa_1x_lr0.04_angle_0_90_180_270',
            'bc_v200.05.01_1x_lr0.04',
            'bc_v200.05.02_1x_lr0.04_IRA',
            'bc_v200.05.03_2x_lr0.04_IRA',
            'bc_v200.05.04_4x_lr0.04_IRA',
            'bc_v200.05.05_1x_lr0.04_FOA',
            'bc_v200.05.06_1x_lr0.04_IRA_FOA',
            'bc_v200.05.07_2x_lr0.04_IRA_FOA',
            'bc_v200.05.08_4x_lr0.04_IRA_FOA',
            'bc_v200.06.01_1x_lr0.04_FOA_no_share',
            'bc_v200.06.02_1x_lr0.04_FOA_share_conv',
            'bc_v200.06.03_1x_lr0.04_FOA_share_conv_fc',
            'bc_v100.01.42_offset_rcnn_r50_1x_public_20201028_lr_0.02_panet',
            'bc_v100.01.43_offset_rcnn_r50_1x_public_20201028_lr_0.02_cascade_mask_rcnn',
            'bc_v100.01.44_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_cascade_mask_rcnn',
            'bc_v100.01.45_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline_pafpn',
            'bc_v100.01.46_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_mask_baseline',
            'bc_v100.01.47_offset_rcnn_r50_2x_public_20201028_building_bbox_footprint_hrnet_baseline',
            'bc_v100.01.48_offset_rcnn_r50_2x_public_20201028_lr_0.02_without_image_rotation_roof_bbox',
            'bc_v100.02.18_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_without_image_rotation_roof',
            'bc_v100.02.19_offset_rcnn_r50_2x_public_20201028_rotate_offset_2_angles_without_rotation_0_90',
            'bc_v100.02.20_offset_rcnn_r50_2x_public_20201028_rotate_offset_3_angles_without_rotation_0_90_180'
            ]

class CSV2Json():
    def __init__(self, ann_file, csv_file, json_prefix):
        self.json_prefix = json_prefix
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(cat_names=('building',))
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
                   metric=['boundary'],
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
        '--city',
        type=str,
        default='shanghai_xian_public', 
        help='dataset for evaluation')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    models = [model for model in ALL_MODELS[0:] if args.version in model]

    print("args.city: ", args.city)
    if args.city == '':
        cities = ['shanghai_xian_public']
    else:
        cities = [args.city]
    
    csv_info = 'splitted'

    for model in models:
        version = model.split('_')[1]

        for city in cities:
            print(f"========== {model} ========== {city} ==========")
            
            if city == 'xian_public':
                anno_file = f'./data/buildchange/public/20201028/coco/annotations/buildchange_public_20201028_val_xian_fine.json'
            elif city == 'shanghai_public':
                anno_file = f'./data/buildchange/public/20201028/coco/annotations/buildchange_public_20201028_val_shanghai_fine_minarea_500.json'
            elif city == 'shanghai_xian_public':
                anno_file = f'./data/buildchange/public/20201028/coco/annotations/buildchange_public_20201028_val_shanghai_xian_minarea_500.json'
                image_dir = f'./data/buildchange/public/20201028/shanghai_xian/images'
            else:
                raise NotImplementedError("do not support city: ", city)
            
            footprint_csv_file = f'../mmdetv2-bc/results/buildchange/{model}/{model}_{city}_footprint_{csv_info}.csv'
            footprint_json_prefix = f'../mmdetv2-bc/results/buildchange/{model}/{model}_{city}_footprint_{csv_info}'

            summary_file = f'./data/buildchange/summary/{model}/{model}_{city}_eval_summary_{csv_info}_AP50.csv'
            csv2json_core = CSV2Json(ann_file=anno_file, 
                                     csv_file=footprint_csv_file, 
                                     json_prefix=footprint_json_prefix)
            
            eval_results = csv2json_core.evaluation()

            print(eval_results, type(eval_results))
            pandas.DataFrame([eval_results]).to_csv(summary_file)