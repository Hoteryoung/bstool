import os
import cv2
import numpy as np
import geopandas
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils
import tqdm
from terminaltables import AsciiTable
from shapely import affinity
import math

import bstool


# matplotlib.use('Agg')

class Evaluation():
    def __init__(self,
                 model=None,
                 anno_file=None,
                 pkl_file=None,
                 gt_roof_csv_file=None,
                 gt_footprint_csv_file=None,
                 roof_csv_file=None,
                 rootprint_csv_file=None,
                 json_prefix=None,
                 iou_threshold=0.1,
                 score_threshold=0.4,
                 min_area=500,
                 with_offset=False,
                 with_height=False,
                 output_dir=None,
                 out_file_format='png',
                 show=True,
                 replace_pred_roof=False,
                 replace_pred_offset=False,
                 with_only_offset=False,
                 offset_model='footprint2roof'):
        self.gt_roof_csv_file = gt_roof_csv_file
        self.gt_footprint_csv_file = gt_footprint_csv_file
        self.roof_csv_file = roof_csv_file
        self.rootprint_csv_file = rootprint_csv_file
        self.pkl_file = pkl_file
        self.json_prefix = json_prefix
        self.show = show
        self.classify_interval=[0,2,4,6,8,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,340,380]
        self.offset_class_num = len(self.classify_interval)
        self.with_only_offset = with_only_offset

        self.out_file_format = out_file_format

        self.output_dir = output_dir
        if output_dir:
            bstool.mkdir_or_exist(self.output_dir)

        if self.with_only_offset:
            pkl_parser = bstool.BSPklParser_Only_Offset(anno_file, 
                                            pkl_file, 
                                            iou_threshold=iou_threshold, 
                                            score_threshold=score_threshold,
                                            min_area=min_area,
                                            with_offset=with_offset, 
                                            with_height=with_height,
                                            gt_roof_csv_file=gt_roof_csv_file,
                                            replace_pred_roof=replace_pred_roof,
                                            offset_model=offset_model)
        else:
            if with_offset:
                pkl_parser = bstool.BSPklParser(anno_file, 
                                            pkl_file, 
                                            iou_threshold=iou_threshold, 
                                            score_threshold=score_threshold,
                                            min_area=min_area,
                                            with_offset=with_offset, 
                                            with_height=with_height,
                                            gt_roof_csv_file=gt_roof_csv_file,
                                            replace_pred_roof=replace_pred_roof,
                                            replace_pred_offset=replace_pred_offset,
                                            offset_model=offset_model)
            else:
                pkl_parser = bstool.BSPklParser_Without_Offset(anno_file, 
                                            pkl_file, 
                                            iou_threshold=iou_threshold, 
                                            score_threshold=score_threshold,
                                            min_area=min_area,
                                            with_offset=with_offset, 
                                            with_height=with_height,
                                            gt_roof_csv_file=gt_roof_csv_file,
                                            replace_pred_roof=replace_pred_roof,
                                            offset_model=offset_model)

        merged_objects = pkl_parser.merged_objects
        
        bstool.bs_csv_dump(merged_objects, roof_csv_file, rootprint_csv_file)

    def _csv2json(self, csv_file, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.img_ids = self.coco.get_img_ids()

        csv_parser = bstool.CSVParse(csv_file)

        bbox_json_results = []
        segm_json_results = []
        for idx in tqdm.tqdm(range(len(self.img_ids))):
            img_id = self.img_ids[idx]
            info = self.coco.load_imgs([img_id])[0]
            image_name = bstool.get_basename(info['file_name'])

            objects = csv_parser(image_name)
            
            masks = [obj['mask'] for obj in objects]
            bboxes = [bstool.mask2bbox(mask) for mask in masks]

            for bbox, mask in zip(bboxes, masks):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = bstool.xyxy2xywh(bbox)
                data['score'] = 1.0
                data['category_id'] = self.category_id

                rles = maskUtils.frPyObjects([mask], self.image_size[0], self.image_size[1])
                rle = maskUtils.merge(rles)
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode()
                data['segmentation'] = rle

                bbox_json_results.append(data)
                segm_json_results.append(data)

        return bbox_json_results, segm_json_results

    def dump_json_results(self):
        result_files = dict()

        # result_files['bbox'] = f'{self.json_prefix}.bbox.json'
        # result_files['segm'] = f'{self.json_prefix}.segm.json'
        # mmcv.dump(self.json_results[0], result_files['bbox'])
        # mmcv.dump(self.json_results[1], result_files['segm'])

        # return result_files

    def _coco_eval(self,
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

    def detection(self, city='xian'):
        pass
        # if city == 'xian':
        #     json_results = _csv2json()


    def _get_length_index(self, length):
        for index, value in enumerate(self.classify_interval):
            if length == 0:
                index = 0
                break
            elif index == self.offset_class_num - 1:
                index = self.offset_class_num
                break
            elif length > self.classify_interval[index] and length <= self.classify_interval[index + 1]:
                index = index + 1
                break
            else:
                pass
        
        return index

    def _get_angle_index(self, angle):
        index = int((angle * 180.0 / np.pi + 180) / 20.0)

        index = 0 if index < 0 else index
        index = 17 if index > 17 else index
        
        return index

    def _rectangle2polar(self, offset):
        length = np.sqrt(offset[0] ** 2 + offset[1] ** 2)
        angle = np.arctan2(offset[1], offset[0])
        return length, angle

    def offset_length_classification(self, title='demo', interval=2, bins=15):
        objects = self.get_confusion_matrix_indexes(mask_type='roof')

        dataset_gt_offsets, dataset_pred_offsets = [], []
        for ori_image_name in self.ori_image_name_list:
            if ori_image_name not in objects.keys():
                continue

            dataset_gt_offsets += objects[ori_image_name]['gt_offsets']
            dataset_pred_offsets += objects[ori_image_name]['pred_offsets']

        confusion_matrix = np.zeros((self.offset_class_num + 1, self.offset_class_num + 1))
        length_error = np.zeros(self.offset_class_num + 1)

        for gt_offset, pred_offset in zip(dataset_gt_offsets, dataset_pred_offsets):
            gt_length = self._rectangle2polar(gt_offset)[0]
            gt_index = self._get_length_index(gt_length)
            pred_length = self._rectangle2polar(pred_offset)[0]
            pred_index = self._get_length_index(pred_length)

            confusion_matrix[gt_index, pred_index] += 1
            length_error[gt_index] += abs(gt_length - pred_length)

        confusion_matrix_small = np.zeros((bins, bins))
        length_error_each_class = length_error / confusion_matrix.sum(axis=1)
        np.set_printoptions(precision=2, floatmode='maxprec')
        np.set_printoptions(suppress=True)
        print("Each length class error (full): ", length_error_each_class)

        region_mean = []
        for idx in range(0, self.offset_class_num + 1, 5):
            _ = length_error_each_class[idx : idx + 5]
            print(f"{idx}: ", np.mean(_[~np.isnan(_)]))
            region_mean.append(np.mean(_[~np.isnan(_)]))

        eval_results = {'classify_interval': self.classify_interval,
                         'length_error_each_class': length_error_each_class.tolist(),
                         'region_mean': region_mean}
        

        for i in range(bins):
            for j in range(bins):
                confusion_matrix_small[i, j] = np.sum(confusion_matrix[interval * i : interval * (i + 1), interval * j : interval * (j + 1)])

        if self.show:
            sns.set(rc={'figure.figsize': (15, 15)})
            fig = sns.heatmap(confusion_matrix_small, annot=True,
                            fmt='g', cmap='PuRd').get_figure()

            fig.savefig(os.path.join(self.output_dir, '{}_offset_confusion_matrix_length_num.{}'.format(title, self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

            fig.clf()
            confusion_matrix_small /= confusion_matrix_small.sum(axis=1)[:, np.newaxis]
            fig = sns.heatmap(confusion_matrix_small, annot=True,
                            fmt='.2f', cmap='PuRd').get_figure()

            fig.savefig(os.path.join(self.output_dir, '{}_offset_confusion_matrix_length_probability.{}'.format(title, self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

            plt.clf()

        return eval_results

    def offset_angle_classification(self, title='demo', interval=1, bins=18):
        objects = self.get_confusion_matrix_indexes(mask_type='roof')

        dataset_gt_offsets, dataset_pred_offsets = [], []
        for ori_image_name in self.ori_image_name_list:
            if ori_image_name not in objects.keys():
                continue

            dataset_gt_offsets += objects[ori_image_name]['gt_offsets']
            dataset_pred_offsets += objects[ori_image_name]['pred_offsets']

        confusion_matrix = np.zeros((18, 18))
        angle_error = np.zeros(18)

        for gt_offset, pred_offset in zip(dataset_gt_offsets, dataset_pred_offsets):
            gt_angle = self._rectangle2polar(gt_offset)[1]
            gt_index = self._get_angle_index(gt_angle)
            pred_angle = self._rectangle2polar(pred_offset)[1]
            pred_index = self._get_angle_index(pred_angle)

            confusion_matrix[gt_index, pred_index] += 1
            angle_error[gt_index] += abs(gt_angle - pred_angle)

        confusion_matrix_small = np.zeros((bins, bins))
        angle_error_each_class = angle_error / confusion_matrix.sum(axis=1)
        np.set_printoptions(precision=2, floatmode='maxprec')
        np.set_printoptions(suppress=True)
        print("Each angle class error (full): ", angle_error_each_class)

        eval_results = {'angle_error_each_class': angle_error_each_class.tolist()}

        for i in range(bins):
            for j in range(bins):
                confusion_matrix_small[i, j] = np.sum(confusion_matrix[interval * i : interval * (i + 1), interval * j : interval * (j + 1)])

        if self.show:
            sns.set(rc={'figure.figsize': (15, 15)})
            fig = sns.heatmap(confusion_matrix_small, annot=True,
                            fmt='g', cmap='PuRd').get_figure()

            fig.savefig(os.path.join(self.output_dir, '{}_offset_confusion_matrix_angle_num.{}'.format(title, self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

            fig.clf()
            confusion_matrix_small /= confusion_matrix_small.sum(axis=1)[:, np.newaxis]
            confusion_matrix_small[np.isnan(confusion_matrix_small)] = 0
            fig = sns.heatmap(confusion_matrix_small, annot=True,
                            fmt='.2f', cmap='PuRd').get_figure()

            fig.savefig(os.path.join(self.output_dir, '{}_offset_confusion_matrix_angle_probability.{}'.format(title, self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

            fig.clf()

        return eval_results

    def cosine_distance(self, a, b):
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        
        similiarity = (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) / (a_norm * b_norm)
        dist = 1.0 - similiarity
        return dist

    def offset_error_vector(self, title='demo', show_polar=False):
        objects = self.get_confusion_matrix_indexes(mask_type='roof')

        dataset_gt_offsets, dataset_pred_offsets = [], []
        for ori_image_name in self.ori_image_name_list:
            if ori_image_name not in objects.keys():
                continue

            dataset_gt_offsets += objects[ori_image_name]['gt_offsets']
            dataset_pred_offsets += objects[ori_image_name]['pred_offsets']

        dataset_gt_offsets = np.array(dataset_gt_offsets)
        dataset_pred_offsets = np.array(dataset_pred_offsets)

        error_vectors = dataset_gt_offsets - dataset_pred_offsets

        EPE = np.sqrt(error_vectors[..., 0] ** 2 + error_vectors[..., 1] ** 2)
        gt_angle = np.arctan2(dataset_gt_offsets[..., 1], dataset_gt_offsets[..., 0])
        gt_length = np.sqrt(dataset_gt_offsets[..., 1] ** 2 + dataset_gt_offsets[..., 0] ** 2)
        
        pred_angle = np.arctan2(dataset_pred_offsets[..., 1], dataset_pred_offsets[..., 0])
        pred_length = np.sqrt(dataset_pred_offsets[..., 1] ** 2 + dataset_pred_offsets[..., 0] ** 2)

        AE = np.abs(gt_angle - pred_angle)

        aEPE = EPE.mean()
        aAE = AE.mean()

        cos_distance = self.cosine_distance(dataset_gt_offsets, dataset_pred_offsets)
        average_cos_distance = cos_distance.mean()

        print(f"Offset AEPE: {aEPE}, aAE: {aAE}, cos distance: ", {average_cos_distance})

        eval_results = {'aEPE': aEPE,
                        'aAE': aAE}

        if self.show:
            r = gt_length - pred_length
            angle = np.abs((gt_angle - pred_angle))
            max_r = np.percentile(r, 95)
            min_r = np.percentile(r, 0.01)

            fig = plt.figure(figsize=(7, 7))
            ax = plt.gca(projection='polar')
            ax.set_thetagrids(np.arange(0.0, 360.0, 15.0))
            ax.set_thetamin(0.0)
            ax.set_thetamax(360.0)
            ax.set_rgrids(np.arange(min_r, max_r, max_r / 10))
            ax.set_rlabel_position(0.0)
            ax.set_rlim(0, max_r)
            plt.setp(ax.get_yticklabels(), fontsize=6)
            ax.grid(True, linestyle = "-", color = "k", linewidth = 0.5, alpha = 0.5)
            ax.set_axisbelow('True')

            plt.scatter(angle, r, s = 2.0)
            plt.title(title + ' offset error distribution', fontsize=10)

            plt.savefig(os.path.join(self.output_dir, '{}_offset_error_polar_evaluation.{}'.format(title, self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

            plt.clf()
            
            max_r = np.percentile(r, 99.99)
            min_r = np.percentile(r, 0.01)
            plt.hist(r, bins=np.arange(min_r, max_r, (int(max_r) - int(min_r)) // 40), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
            plt.title(title + ' Length Error Distribution', fontsize=10)
            plt.xlim([min_r - 5, max_r + 5])
            plt.xlabel('Error')
            plt.ylabel('Num')
            plt.yscale('log')
            plt.savefig(os.path.join(self.output_dir, '{}_offset_error_length_hist_evaluation.{}'.format(title, self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

            plt.clf()
            
            max_angle = angle.max() * 180.0 / np.pi
            min_angle = angle.min() * 180.0 / np.pi
            plt.hist(r, bins=np.arange(min_angle, max_angle, (max_angle - min_angle) // 80), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
            plt.title(title + ' Angle Error Distribution', fontsize=10)
            plt.xlim([min_angle - 20, max_angle])
            plt.xlabel('Error')
            plt.ylabel('Num')
            plt.yscale('log')
            plt.savefig(os.path.join(self.output_dir, '{}_offset_error_angle_hist_evaluation.{}'.format(title, self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

            plt.clf()

        return eval_results

    def segmentation(self, mask_types = ['roof', 'footprint']):
        eval_results = dict()
        for mask_type in mask_types:
            print(f"========== Processing {mask_type} segmentation ==========")
            objects = self.get_confusion_matrix_indexes(mask_type=mask_type)

            dataset_gt_TP_indexes, dataset_pred_TP_indexes, dataset_gt_FN_indexes, dataset_pred_FP_indexes = [], [], [], []
            for ori_image_name in self.ori_image_name_list:
                if ori_image_name not in objects.keys():
                    continue

                gt_TP_indexes = objects[ori_image_name]['gt_TP_indexes']
                pred_TP_indexes = objects[ori_image_name]['pred_TP_indexes']
                gt_FN_indexes = objects[ori_image_name]['gt_FN_indexes']
                pred_FP_indexes = objects[ori_image_name]['pred_FP_indexes']

                dataset_gt_TP_indexes += gt_TP_indexes
                dataset_pred_TP_indexes += pred_TP_indexes
                dataset_gt_FN_indexes += gt_FN_indexes
                dataset_pred_FP_indexes += pred_FP_indexes

            TP = len(dataset_gt_TP_indexes)
            FN = len(dataset_gt_FN_indexes)
            FP = len(dataset_pred_FP_indexes)

            print("Summary (codes by jwwangchn):")
            print("TP: ", TP)
            print("FN: ", FN)
            print("FP: ", FP)

            Precision = float(TP) / (float(TP) + float(FP))
            Recall = float(TP) / (float(TP) + float(FN))

            F1_score = (2 * Precision * Recall) / (Precision + Recall)
            print("Precision: ", Precision)
            print("Recall: ", Recall)

            print("F1 score: ", F1_score)

            eval_results[mask_type] = {'F1_score': F1_score,
                                        'Precision': Precision,
                                        'Recall': Recall,
                                        'TP': TP,
                                        'FN': FN,
                                        'FP': FP}
        
        return eval_results

    def offset_angle(self, title='dalian'):
        objects = self.get_confusion_matrix_indexes(mask_type='roof')

        errors = []

        dataset_gt_offsets, dataset_pred_heights = [], []
        gt_angle_std = []
        pred_angle_std = []
        for ori_image_name in self.ori_image_name_list:
            if ori_image_name not in objects.keys():
                continue

            dataset_gt_heights += objects[ori_image_name]['gt_heights']
            dataset_pred_heights += objects[ori_image_name]['pred_heights']

            gt_heights = np.array(objects[ori_image_name]['gt_heights'])
            pred_heights = np.array(objects[ori_image_name]['pred_heights'])

            if len(gt_heights) == 0 or len(pred_heights) == 0:
                continue

            max_height = np.percentile(gt_heights, percent)
            bool_keep = gt_heights < max_height
            gt_heights = gt_heights[bool_keep]
            pred_heights = pred_heights[bool_keep]

            if len(gt_heights) == 0 or len(pred_heights) == 0:
                continue

            error = np.abs(gt_heights - pred_heights)
            errors += error.tolist()

            gt_offsets = np.array(objects[ori_image_name]['gt_offsets'])
            pred_offsets = np.array(objects[ori_image_name]['pred_offsets'])

            # gt_avg_offsets = np.array(gt_offsets) / np.array(gt_heights)
            gt_angle = np.arctan2(gt_offsets[:, 1], gt_offsets[:, 0])

            # pred_avg_offsets = np.array(pred_offsets) / np.array(pred_heights)
            pred_angle = np.arctan2(pred_offsets[:, 1], pred_offsets[:, 0])

            single_image_gt_angle_std = gt_angle.std()
            single_image_pred_angle_std = pred_angle.std()

            gt_angle_std.append(single_image_gt_angle_std)
            pred_angle_std.append(single_image_pred_angle_std)

        mse = np.array(errors).mean()

        print("Evaluation pkl file: ", self.pkl_file)
        print("Height MSE: ", mse)
        print("GT angle Std: ", np.array(gt_angle_std).mean())
        print("Pred angle Std: ", np.array(pred_angle_std).mean())

        if self.show:
            y_gt = np.array(dataset_gt_heights)
            y_pred = np.array(dataset_pred_heights)
            
            sort_index = np.argsort(y_gt)[::-1]
            y_gt = y_gt[sort_index]
            y_pred = y_pred[sort_index]

            x = range(y_gt.shape[0])
            
            plt.plot(x, y_gt)
            plt.scatter(x, y_pred, s=1, color='green')
            plt.title(title + ' {}'.format(mse))
            plt.savefig(os.path.join(self.output_dir, '{}_height_evaluation.{}'.format(title, self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

    def height(self, percent=100, title='dalian'):
        objects = self.get_confusion_matrix_indexes(mask_type='roof')

        errors = []

        dataset_gt_heights, dataset_pred_heights = [], []
        gt_angle_std = []
        pred_angle_std = []
        for ori_image_name in self.ori_image_name_list:
            if ori_image_name not in objects.keys():
                continue

            gt_heights = np.array(objects[ori_image_name]['gt_heights'])
            pred_heights = np.array(objects[ori_image_name]['pred_heights'])

            if len(gt_heights) == 0 or len(pred_heights) == 0:
                continue

            bool_keep = gt_heights < 10000
            gt_heights = gt_heights[bool_keep]
            pred_heights = pred_heights[bool_keep]

            dataset_gt_heights += gt_heights.tolist()
            dataset_pred_heights += pred_heights.tolist()

            if len(gt_heights) == 0 or len(pred_heights) == 0:
                continue

            error = np.abs(gt_heights - pred_heights)
            errors += error.tolist()

            gt_offsets = np.array(objects[ori_image_name]['gt_offsets'])
            pred_offsets = np.array(objects[ori_image_name]['pred_offsets'])

            # gt_avg_offsets = np.array(gt_offsets) / np.array(gt_heights)
            gt_angle = np.arctan2(gt_offsets[:, 1], gt_offsets[:, 0])

            # pred_avg_offsets = np.array(pred_offsets) / np.array(pred_heights)
            pred_angle = np.arctan2(pred_offsets[:, 1], pred_offsets[:, 0])

            single_image_gt_angle_std = gt_angle.std()
            single_image_pred_angle_std = pred_angle.std()

            gt_angle_std.append(single_image_gt_angle_std)
            pred_angle_std.append(single_image_pred_angle_std)

        mse = np.array(errors).mean()

        print("Evaluation pkl file: ", self.pkl_file)
        print("Height MSE: ", mse)
        print("GT angle Std: ", np.array(gt_angle_std).mean())
        print("Pred angle Std: ", np.array(pred_angle_std).mean())

        if self.show:
            plt.clf()
            
            y_gt = np.array(dataset_gt_heights)
            y_pred = np.array(dataset_pred_heights)
            
            sort_index = np.argsort(y_gt)[::-1]
            y_gt = y_gt[sort_index]
            y_pred = y_pred[sort_index]

            x = range(y_gt.shape[0])
            
            plt.plot(x, y_gt)
            plt.scatter(x, y_pred, s=1, color='green')
            plt.title(title + ' {}'.format(mse))
            plt.xlabel('index')
            plt.ylabel('height (m)')
            plt.savefig(os.path.join(self.output_dir, '{}_height_evaluation.{}'.format(title, self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

    def get_confusion_matrix_indexes(self, mask_type='footprint'):
        if mask_type == 'footprint':
            gt_csv_parser = bstool.CSVParse(self.gt_footprint_csv_file)
            pred_csv_parser = bstool.CSVParse(self.rootprint_csv_file)
        else:
            gt_csv_parser = bstool.CSVParse(self.gt_roof_csv_file)
            pred_csv_parser = bstool.CSVParse(self.roof_csv_file)

        self.ori_image_name_list = gt_csv_parser.image_name_list

        gt_objects = gt_csv_parser.objects
        pred_objects = pred_csv_parser.objects

        objects = dict()

        for ori_image_name in self.ori_image_name_list:
            buildings = dict()
            
            gt_buildings = gt_objects[ori_image_name]
            pred_buildings = pred_objects[ori_image_name]

            gt_polygons = [gt_building['polygon'] for gt_building in gt_buildings]
            pred_polygons = [pred_building['polygon'] for pred_building in pred_buildings]

            gt_polygons_origin = gt_polygons[:]
            pred_polygons_origin = pred_polygons[:]
            
            if len(gt_polygons) == 0 or len(pred_polygons) == 0:
                continue

            gt_offsets = [gt_building['offset'] for gt_building in gt_buildings]
            pred_offsets = [pred_building['offset'] for pred_building in pred_buildings]
            
            gt_heights = [gt_building['height'] for gt_building in gt_buildings]
            pred_heights = [pred_building['height'] for pred_building in pred_buildings]

            angles = []
            for gt_offset, gt_height in zip(gt_offsets, gt_heights):
                offset_x, offset_y = gt_offset
                angle = math.atan2(math.sqrt(offset_x ** 2 + offset_y ** 2) * 0.6, gt_height)
                angles.append(angle)

            height_angle = np.array(angles).mean()

            gt_polygons = geopandas.GeoSeries(gt_polygons)
            pred_polygons = geopandas.GeoSeries(pred_polygons)

            gt_df = geopandas.GeoDataFrame({'geometry': gt_polygons, 'gt_df':range(len(gt_polygons))})
            pred_df = geopandas.GeoDataFrame({'geometry': pred_polygons, 'pred_df':range(len(pred_polygons))})

            gt_df = gt_df.loc[~gt_df.geometry.is_empty]
            pred_df = pred_df.loc[~pred_df.geometry.is_empty]

            res_intersection = geopandas.overlay(gt_df, pred_df, how='intersection')

            iou = np.zeros((len(pred_polygons), len(gt_polygons)))
            for idx, row in res_intersection.iterrows():
                gt_idx = row.gt_df
                pred_idx = row.pred_df

                inter = row.geometry.area
                union = pred_polygons[pred_idx].area + gt_polygons[gt_idx].area

                iou[pred_idx, gt_idx] = inter / (union - inter + 1.0)

            iou_indexes = np.argwhere(iou >= 0.5)

            gt_TP_indexes = list(iou_indexes[:, 1])
            pred_TP_indexes = list(iou_indexes[:, 0])

            gt_FN_indexes = list(set(range(len(gt_polygons))) - set(gt_TP_indexes))
            pred_FP_indexes = list(set(range(len(pred_polygons))) - set(pred_TP_indexes))

            buildings['gt_iou'] = np.max(iou, axis=0)

            buildings['gt_TP_indexes'] = gt_TP_indexes
            buildings['pred_TP_indexes'] = pred_TP_indexes
            buildings['gt_FN_indexes'] = gt_FN_indexes
            buildings['pred_FP_indexes'] = pred_FP_indexes

            buildings['gt_offsets'] = np.array(gt_offsets)[gt_TP_indexes].tolist()
            buildings['pred_offsets'] = np.array(pred_offsets)[pred_TP_indexes].tolist()

            buildings['gt_heights'] = np.array(gt_heights)[gt_TP_indexes].tolist()
            buildings['pred_heights'] = np.array(pred_heights)[pred_TP_indexes].tolist()

            buildings['gt_polygons'] = gt_polygons
            buildings['pred_polygons'] = pred_polygons

            buildings['gt_polygons_matched'] = np.array(gt_polygons_origin)[gt_TP_indexes].tolist()
            buildings['pred_polygons_matched'] = np.array(pred_polygons_origin)[pred_TP_indexes].tolist()

            buildings['height_angle'] = height_angle

            objects[ori_image_name] = buildings

        return objects

    def visualization_boundary(self, image_dir, vis_dir, mask_types=['roof', 'footprint'], with_iou=False):
        colors = {'gt_TP':   (0, 255, 0),
                'pred_TP': (255, 255, 0),
                'FP':      (0, 255, 255),
                'FN':      (255, 0, 0)}
        for mask_type in mask_types:
            objects = self.get_confusion_matrix_indexes(mask_type=mask_type)

            for image_name in os.listdir(image_dir):
                image_basename = bstool.get_basename(image_name)
                image_file = os.path.join(image_dir, image_name)

                output_file = os.path.join(vis_dir, mask_type, image_name)
                bstool.mkdir_or_exist(os.path.join(vis_dir, mask_type))

                img = cv2.imread(image_file)

                if image_basename not in objects:
                    continue

                building = objects[image_basename]

                for idx, gt_polygon in enumerate(building['gt_polygons']):
                    iou = building['gt_iou'][idx]
                    if idx in building['gt_TP_indexes']:
                        color = colors['gt_TP'][::-1]
                    else:
                        color = colors['FN'][::-1]

                    if gt_polygon.geom_type != 'Polygon':
                        continue

                    img = bstool.draw_mask_boundary(img, bstool.polygon2mask(gt_polygon), color=color)
                    if with_iou:
                        img = bstool.draw_iou(img, gt_polygon, iou, color=color)

                for idx, pred_polygon in enumerate(building['pred_polygons']):
                    if idx in building['pred_TP_indexes']:
                        color = colors['pred_TP'][::-1]
                    else:
                        color = colors['FP'][::-1]

                    if gt_polygon.geom_type != 'Polygon':
                        continue

                    img = bstool.draw_mask_boundary(img, bstool.polygon2mask(pred_polygon), color=color)

                cv2.imwrite(output_file, img)

    def visualization_offset(self, image_dir, vis_dir, with_footprint=True):
        print("========== generation vis images with offset ==========")
        if with_footprint:
            image_dir = os.path.join(vis_dir, '..', 'boundary', 'footprint')
            vis_dir = vis_dir + '_with_footprint'
            
        bstool.mkdir_or_exist(vis_dir)

        colors = {'gt_matched':   (0, 255, 0),
                  'pred_matched': (255, 255, 0),
                  'pred_un_matched': (0, 255, 255),
                  'gt_un_matched': (255, 0, 0)}
        objects = self.get_confusion_matrix_indexes(mask_type='roof')

        for image_name in os.listdir(image_dir):
            image_basename = bstool.get_basename(image_name)
            image_file = os.path.join(image_dir, image_name)

            output_file = os.path.join(vis_dir, image_name)

            img = cv2.imread(image_file)

            if image_basename not in objects:
                continue

            building = objects[image_basename]

            height_angle = building['height_angle']

            img = bstool.draw_height_angle(img, height_angle)

            for gt_polygon, gt_offset, pred_polygon, pred_offset, gt_height in zip(building['gt_polygons_matched'], building['gt_offsets'], building['pred_polygons_matched'], building['pred_offsets'], building['gt_heights']):
                gt_roof_centroid = list(gt_polygon.centroid.coords)[0]
                pred_roof_centroid = list(pred_polygon.centroid.coords)[0]

                gt_footprint_centroid = [coordinate - offset for coordinate, offset in zip(gt_roof_centroid, gt_offset)]
                pred_footprint_centroid = [coordinate - offset for coordinate, offset in zip(pred_roof_centroid, pred_offset)]

                xoffset, yoffset = gt_offset
                transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
                gt_footprint_polygon = affinity.affine_transform(gt_polygon, transform_matrix)

                xoffset, yoffset = pred_offset
                transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
                pred_footprint_polygon = affinity.affine_transform(pred_polygon, transform_matrix)

                intersection = gt_footprint_polygon.intersection(pred_footprint_polygon).area
                union = gt_footprint_polygon.union(pred_footprint_polygon).area

                iou = intersection / (union - intersection + 1.0)

                if iou >= 0.5:
                    gt_color = colors['gt_matched'][::-1]
                    pred_color = colors['pred_matched'][::-1]
                else:
                    gt_color = colors['gt_un_matched'][::-1]
                    pred_color = colors['pred_un_matched'][::-1]

                img = bstool.draw_offset_arrow(img, gt_roof_centroid, gt_footprint_centroid, color=gt_color)
                img = bstool.draw_offset_arrow(img, pred_roof_centroid, pred_footprint_centroid, color=pred_color)

            cv2.imwrite(output_file, img)