import cv2
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import data, segmentation, color

from skimage.future import graph

import bstool


def grayscale_5_levels(gray):
    level_list = [[200, 255], [175, 224], [150, 199], [125, 174], [100, 149], [75, 124], [50, 99], [25, 74], [1, 49]]
    # level_list = [[200, 255], [150, 199], [100, 149], [50, 99], [1, 49]]
    # level_list = [[170, 255], [80, 169], [1, 79]]
    high = 255
    objects = []
    for level in level_list:
        low, high = level

        # low = high - 51
        # col_to_be_changed_low = np.array([low])
        # col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(gray, low, high)

        # print(curr_mask, curr_mask.max(), curr_mask.min())
        # gray[curr_mask > 0] = high

        # bstool.show_image(curr_mask)

        single_level_objects = bstool.mask_parse(curr_mask.copy(), (255, 255))

        # print(len(single_level_objects))

        # polygons = [obj['polygon'] for obj in single_level_objects]

        # bstool.show_polygons_on_image(rgb_img, polygons)

        objects += single_level_objects
        
        # high -= 51
        # if low <= 0:
        #     break

    return objects

if __name__ == '__main__':
    idx = '208'
    AGL_file = '/data/urban3d/JAX_VAL/Val/001/JAX_Tile_{}_AGL_001.tif'.format(idx)
    FACADE_file = '/data/urban3d/JAX_VAL/Val/001/JAX_Tile_{}_FACADE_001.tif'.format(idx)
    RGB_file = '/data/urban3d/JAX_VAL/Val/001/JAX_Tile_{}_RGB_001.tif'.format(idx)

    AGL_img = rio.open(AGL_file)
    height = AGL_img.read(1)

    FACADE_img = rio.open(FACADE_file)
    facade = FACADE_img.read(1)

    rgb_img = cv2.imread(RGB_file)
    
    roof = bstool.generate_subclass_mask(facade, (6, 6,))

    masked_height = (height + 1.0 + height.min()) * roof

    # plt.imshow(roof)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # plt.imshow(height)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    img = (masked_height - masked_height.min()) / (masked_height.max() - masked_height.min()) * 255
    img = img.astype(np.uint8)

    # plt.imshow(img)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    objects = grayscale_5_levels(img)

    polygons = [obj['polygon'] for obj in objects]
    masks = [obj['mask'] for obj in objects]

    bstool.show_polygons_on_image(rgb_img, polygons)