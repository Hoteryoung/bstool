# -*- encoding: utf-8 -*-
'''
@File    :   vis_image_translation.py
@Time    :   2020/12/30 22:06:41
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2020
@Desc    :   可视化图像变换
'''


import cv2

import bstool

if __name__ == '__main__':
    image_file = '/data/plane/v0/train/images/1.tif'
    img = cv2.imread(image_file)

    img = bstool.image_translation(img, 200, -200)

    bstool.show_image(img)