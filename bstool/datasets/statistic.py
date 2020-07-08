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

import bstool


matplotlib.use('Agg')

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
                    building['height'] = row.Height
                else:
                    building['height'] = 0

                buildings.append(building)

            objects += buildings

        return objects

    def height(self):
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
        
        plt.hist(heights, bins=np.arange(0, 100, 100 / 30), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
        plt.savefig(os.path.join(self.output_dir, f'height.{self.out_file_format}'), bbox_inches='tight', dpi=600, pad_inches=0.1)
