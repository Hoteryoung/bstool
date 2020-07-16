import os
import numpy as np
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from collections import  Counter
from collections import defaultdict
from shapely.geometry import Polygon
import pandas
import matplotlib
import math
import ast

import bstool


# matplotlib.use('Agg')

# plt.rcParams.update({'font.size': 14})    # ICPR paper
plt.rcParams.update({'font.size': 12})

# plt.rcParams["font.family"] = "Times New Roman"

class Statistic():
    def __init__(self, 
                 ann_file=None, 
                 csv_file=None,
                 output_dir='./data/buildchange/v0/statistic',
                 out_file_format='png'):
        bstool.mkdir_or_exist(output_dir)
        self.output_dir = output_dir
        self.out_file_format = out_file_format

        if isinstance(csv_file, str):
            self.objects = self._parse_csv(csv_file)
        elif isinstance(csv_file, list):
            self.objects = []
            for csv_file_ in csv_file:
                print("Processing: ", csv_file_)
                self.objects += self._parse_csv(csv_file_)

    def _parse_coco(self, ann_file):
        coco =  COCO(ann_file)
        img_ids = coco.get_img_ids()

        objects = []

        for idx, img_id in enumerate(img_ids):
            buildings = []
            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            anns = coco.load_anns(ann_ids)

            building = dict()
            for ann in anns:
                building['height'] = ann['building_height']
                buildings.append(building)

            objects += buildings

        return objects

    def _parse_csv(self, csv_file):
        csv_df = pandas.read_csv(csv_file)
        image_name_list = list(set(csv_df.ImageId.unique()))

        objects = []
        for image_name in image_name_list:
            buildings = []
            for idx, row in csv_df[csv_df.ImageId == image_name].iterrows():
                building = dict()
                obj_keys = row.to_dict().keys()

                if 'Height' in obj_keys:
                    building['height'] = int(row.Height)
                else:
                    building['height'] = 3.0

                if 'Offset' in obj_keys:
                    if type(row.Offset) == str:
                        building['offset'] = ast.literal_eval(row.Offset)
                    else:
                        building['offset'] = row.Offset
                else:
                    building['offset'] = [0.0, 0.0]

                buildings.append(building)
            objects += buildings

        return objects

    def height_distribution(self, title=['all']):
        heights = np.array([obj['height'] for obj in self.objects])

        print("Height mean: ", heights.mean())
        print("Height std: ", heights.std())
        height_90 = np.percentile(heights, 90)
        height_80 = np.percentile(heights, 80)
        height_70 = np.percentile(heights, 70)
        height_60 = np.percentile(heights, 60)

        print("Height 90: ", heights[heights < height_90].mean(), heights[heights < height_90].std())
        print("Height 80: ", heights[heights < height_80].mean(), heights[heights < height_80].std())
        print("Height 70: ", heights[heights < height_70].mean(), heights[heights < height_70].std())
        print("Height 60: ", heights[heights < height_60].mean(), heights[heights < height_60].std())
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.hist(heights, bins=np.arange(0, 100, 100 / 30), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
        ax.set_yscale('log', basey=10)
        plt.title("_".join(title))
        plt.xlabel('height (m)')
        plt.ylabel('count')
        plt.savefig(os.path.join(self.output_dir, '{}_height.{}'.format("_".join(title), self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

        plt.clf()

    def height_curve(self, title=['all']):
        heights = np.array([obj['height'] for obj in self.objects])

        heights = np.sort(heights)[::-1]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        x = range(heights.shape[0])
        
        ax.plot(x, heights)
        plt.title("_".join(title))
        plt.xlabel('index')
        plt.ylabel('height (m)')
        plt.savefig(os.path.join(self.output_dir, '{}_height_curve.{}'.format("_".join(title), self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

        plt.clf()

    def offset_polar(self, title=['all']):
        offsets = np.array([obj['offset'] for obj in self.objects])

        r = np.sqrt(offsets[:, 0] ** 2 + offsets[:, 1] ** 2)
        angle = np.arctan2(offsets[:, 1], offsets[:, 0]) * 180.0 / np.pi
        max_r = np.percentile(r, 90)

        fig = plt.figure(figsize=(7, 7))
        ax = plt.gca(projection='polar')
        ax.set_thetagrids(np.arange(0.0, 360.0, 15.0))
        ax.set_thetamin(0.0)
        ax.set_thetamax(360.0)
        ax.set_rgrids(np.arange(0, max_r, max_r / 10))
        ax.set_rlabel_position(0.0)
        ax.set_rlim(0, max_r)
        plt.setp(ax.get_yticklabels(), fontsize=6)
        ax.grid(True, linestyle = "-", color = "k", linewidth = 0.5, alpha = 0.5)
        ax.set_axisbelow('True')

        plt.scatter(angle, r, s = 2.0)
        plt.title("_".join(title) + ' offset polar distribution', fontsize=10)

        plt.savefig(os.path.join(self.output_dir, '{}_offset_polar.{}'.format("_".join(title), self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

        plt.clf()

    def offset_distribution(self, title=['all']):
        offsets = np.array([obj['offset'] for obj in self.objects])

        print("Offset X mean: ", offsets[:, 0].mean())
        print("Offset X std: ", offsets[:, 0].std())
        print("Offset X median: ", np.median(offsets[:, 0]))

        print("Offset Y mean: ", offsets[:, 1].mean())
        print("Offset Y std: ", offsets[:, 1].std())
        print("Offset Y median: ", np.median(offsets[:, 1]))
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        ax[0].hist(offsets[:, 0], bins=np.arange(-20, 20, 40 / 200), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
        # ax.set_yscale('log', basey=10)
        ax[0].set_title("_".join(title) + ' offset X distribution', fontsize=10)
        ax[0].set_xlabel('offset (pixel)')
        ax[0].set_ylabel('count')

        ax[1].hist(offsets[:, 1], bins=np.arange(-20, 20, 40 / 100), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
        # ax.set_yscale('log', basey=10)
        ax[1].set_title("_".join(title) + ' offset Y distribution', fontsize=10)
        ax[1].set_xlabel('offset (pixel)')
        ax[1].set_ylabel('count')

        plt.savefig(os.path.join(self.output_dir, '{}_offset_distribution.{}'.format("_".join(title), self.out_file_format)), bbox_inches='tight', dpi=600, pad_inches=0.1)

        plt.clf()