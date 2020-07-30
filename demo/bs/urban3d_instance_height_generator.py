import cv2
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import data, segmentation, color

from skimage.future import graph

import bstool


def grayscale_5_levels(gray):
    high = 255
    while(1):  
        low = high - 51
        col_to_be_changed_low = np.array([low])
        col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(gray, col_to_be_changed_low, col_to_be_changed_high)
        gray[curr_mask > 0] = (high)
        high -= 51
        if(low <= 0):
            break

if __name__ == '__main__':
    AGL_file = '/data/urban3d/JAX_VAL/Val/001/JAX_Tile_208_AGL_001.tif'
    FACADE_file = '/data/urban3d/JAX_VAL/Val/001/JAX_Tile_208_FACADE_001.tif'

    AGL_img = rio.open(AGL_file)
    height = AGL_img.read(1)

    FACADE_img = rio.open(FACADE_file)
    facade = FACADE_img.read(1)
    
    roof = bstool.generate_subclass_mask(facade, (6, 6,))

    masked_height = height * roof

    img = (masked_height - masked_height.min()) / (masked_height.max() - masked_height.min()) * 255
    img = img.astype(np.uint8)

    grayscale_5_levels(img)

    plt.imshow(img)
    plt.show()









    # img = (masked_height - masked_height.min()) / (masked_height.max() - masked_height.min()) * 255

    # img = img.astype(np.uint8)
    # img = color.gray2rgb(img)

    # # img = data.coffee()

    # print(img.shape, img.dtype, img.max(), img.min())

    # labels1 = segmentation.slic(img, compactness=10, n_segments=100)
    # out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

    # g = graph.rag_mean_color(img, labels1)
    # labels2 = graph.cut_threshold(labels1, g, 29)
    # out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

    # fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
    #                     figsize=(6, 8))

    # ax[0].imshow(out1)
    # ax[1].imshow(out2)

    # for a in ax:
    #     a.axis('off')

    # plt.tight_layout()

    # plt.show()