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

gt_length = np.array([1,2,3,4,5,6,7,8,9])
pred_length = np.array([2,4,3,5,4,7,8,1,9])

gt_angle = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
pred_angle = np.array([0.2, 0.4, 0.3, 0.5, 0.4, 0.7, 0.8, 0.1, 0.9])

# r = np.abs(gt_length - pred_length)
# angle = ((gt_angle - pred_angle) % np.pi) * 180.0 / np.pi
r = 10
angle = -np.pi / 4
# max_r = np.percentile(r, 95)

max_r = 15

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
plt.title('' + ' offset error distribution', fontsize=10)

plt.show()