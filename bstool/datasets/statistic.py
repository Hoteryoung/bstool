import numpy as np
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import cm
from collections import defaultdict
from shapely.geometry import Polygon

import wwtool

# plt.rcParams.update({'font.size': 14})    # ICPR paper
plt.rcParams.update({'font.size': 12})

# plt.rcParams["font.family"] = "Times New Roman"

class COCO_Statistic():
    def __init__(self, 
                ann_file, 
                size_set=[16*16, 32*32, 96*96], 
                label_set=[], 
                size_measure_by_ratio=False,
                class_instance=None,
                show_title=False,
                out_file_format='pdf',
                max_object_num=2700,
                min_area=0,
                max_small_length=0):
        self.ann_file = ann_file
        self.coco = COCO(self.ann_file)
        self.catIds = self.coco.getCatIds(catNms=[''])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.image_num = len(self.imgIds)
        self.size_set = size_set
        self.label_set = label_set
        self.size_measure_by_ratio = size_measure_by_ratio
        self.class_instance = class_instance
        self.show_title = show_title
        self.out_file_format = out_file_format
        self.max_object_num = max_object_num
        self.min_area = min_area
        self.max_small_length = max_small_length

        categories = self.coco.dataset['categories']
        self.coco_class = dict()
        for category in categories:
            self.coco_class[category['id']] = category['name']
