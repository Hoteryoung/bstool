import cv2
import os
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import data, segmentation, color, feature
from skimage.morphology import square, dilation, erosion, closing, opening
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.restoration import inpaint
from skimage.future import graph
import skimage
import shapely
import itertools

import bstool


def merge_contain_polygon(polygons):
    keep_idx = list(range(len(polygons)))
    
    polygon_combinnations = list(itertools.combinations(range(len(polygons)), 2))
    drop_idx = []
    for combination in polygon_combinnations:
        polygon_a, polygon_b = polygons[combination[0]], polygons[combination[1]]
        if polygon_a.contains(polygon_b):
            # if polygon_a.geom_type == 'MultiPolygon':
            #     continue
            drop_idx.append(combination[1])

    remain_idx = [_ for _ in keep_idx if _ not in drop_idx]

    polygons = np.array(polygons)[remain_idx].tolist()

    return polygons

def mask2polygons(mask, edge, min_area=100):
    result = bstool.generate_image(mask.shape[0], mask.shape[1], 0)
    keep = np.logical_and(mask == 1, edge == False)
    
    result[keep] = 1
    objects = bstool.mask_parse(result)

    polygons = []
    for obj in objects:
        polygon = obj['polygon']
        if polygon.area < min_area:
            continue

        polygons.append(polygon)

    print("before: ", len(polygons))
    polygons = merge_contain_polygon(polygons)
    print("after: ", len(polygons))

    return polygons

def mask2edge(rgb, roof_mask, mask_height, save_dir, square_size=7, sigma=5, with_dilation=True, with_bilateral=False, show_compare=False, show_final=True):
    input_mask_height = mask_height.copy()
    if with_bilateral:
        mask_height = denoise_bilateral(mask_height, sigma_color=1.0, sigma_spatial=9)

    if with_dilation:
        mask_height = dilation(mask_height, square(square_size))
    
    edges = feature.canny(mask_height, sigma=sigma)

    polygons = mask2polygons(roof_mask, edges)

    if show_compare:
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        ax1, ax2, ax3, ax4 = ax[0][0], ax[0][1], ax[1][0], ax[1][1]

        ax1.imshow(rgb)
        ax1.axis('off')
        ax1.set_title('rgb')

        ax2.imshow(input_mask_height, cmap=plt.cm.jet)
        ax2.axis('off')
        ax2.set_title('roof depth')

        ax3.imshow(edges, cmap=plt.cm.gray)
        ax3.axis('off')
        ax3.set_title('edge')

        ax4.imshow(bstool.generate_image(rgb.shape[0], rgb.shape[1], 255))
        ax4.axis('off')
        ax4.set_title('polygons')

        for polygon in polygons:
            if type(polygon) == str:
                polygon = shapely.wkt.loads(polygon)

            ax4.plot(*polygon.exterior.xy)
        
        # plt.savefig(os.path.join(save_dir, f'{idx}_{sigma}.png'), bbox_inches='tight', dpi=600, pad_inches=0.1)

        # plt.show()
        plt.close()

    if show_final:
        plt.imshow(rgb)
        plt.axis('off')

        for polygon in polygons:
            if type(polygon) == str:
                polygon = shapely.wkt.loads(polygon)

            plt.plot(*polygon.exterior.xy)

        plt.savefig(os.path.join(save_dir, f'{idx}_canny.png'), bbox_inches='tight', dpi=600, pad_inches=0.1)
        plt.show()


if __name__ == '__main__':
    save_dir = '/data/urban3d/vis'
    root_dir = '/data/urban3d/JAX_VAL/Val/001'
    file_names = os.listdir(root_dir)
    
    with_norm = False
    
    indexes = []
    for file_name in file_names:
        if not file_name.endswith('.tif'):
            continue
        idx = file_name.split("_")[2]
        indexes.append(idx)
    indexes = list(set(indexes))

    for idx in indexes:
        AGL_file = '/data/urban3d/JAX_VAL/Val/001/JAX_Tile_{}_AGL_001.tif'.format(idx)
        FACADE_file = '/data/urban3d/JAX_VAL/Val/001/JAX_Tile_{}_FACADE_001.tif'.format(idx)
        RGB_file = '/data/urban3d/JAX_VAL/Val/001/JAX_Tile_{}_RGB_001.tif'.format(idx)

        AGL_img = rio.open(AGL_file)
        height = AGL_img.read(1)

        FACADE_img = rio.open(FACADE_file)
        facade = FACADE_img.read(1)

        rgb = cv2.imread(RGB_file)
        
        roof_mask = bstool.generate_subclass_mask(facade, (6, 6,))
        masked_height = height * roof_mask
        masked_height[np.isnan(masked_height)] = 0.0

        height_99 = np.percentile(masked_height, 99.9999)
        height_01 = np.percentile(masked_height, 0.0001)

        masked_height[masked_height > height_99] = 0.0
        masked_height[masked_height < height_01] = 0.0

        if with_norm:
            masked_height = (masked_height - masked_height.min()) / (masked_height.max() - masked_height.min()) * 255
            masked_height = masked_height.astype(np.uint8)

        mask2edge(rgb, roof_mask, masked_height, save_dir)