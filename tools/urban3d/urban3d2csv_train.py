import os
import cv2
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import skimage
import shapely
import itertools

import mmcv
import bstool


def mask2polygons(rgb, masked_height, roof_mask, show_compare=False, show_final=True, min_area=100):

    objects = bstool.mask_parse(roof_mask)

    polygons = []
    for obj in objects:
        polygon = obj['polygon']
        if polygon.area < min_area:
            continue

        polygons.append(polygon)

    if show_compare:
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        ax1, ax2, ax3, ax4 = ax[0][0], ax[0][1], ax[1][0], ax[1][1]

        ax1.imshow(rgb)
        ax1.axis('off')
        ax1.set_title('rgb')

        ax2.imshow(masked_height, cmap=plt.cm.jet)
        ax2.axis('off')
        ax2.set_title('roof depth')

        ax3.imshow(roof_mask, cmap=plt.cm.gray)
        ax3.axis('off')
        ax3.set_title('roof mask')

        ax4.imshow(rgb)
        ax4.axis('off')
        ax4.set_title('polygons')

        for polygon in polygons:
            if type(polygon) == str:
                polygon = shapely.wkt.loads(polygon)

            ax4.plot(*polygon.exterior.xy, linewidth=3)

        # plt.show()
        plt.close()

    if show_final:
        plt.imshow(rgb)
        plt.axis('off')

        for polygon in polygons:
            if type(polygon) == str:
                polygon = shapely.wkt.loads(polygon)

            plt.plot(*polygon.exterior.xy)

        plt.savefig(os.path.join(save_dir, f'{idx}_offset.png'), bbox_inches='tight', dpi=600, pad_inches=0.1)
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
        VFLOW_file = '/data/urban3d/JAX_VAL/Val/001/JAX_Tile_{}_VFLOW_001.json'.format(idx)
        BLDG_FTPRINT_file = '/data/urban3d/JAX_VAL/Val/001/JAX_Tile_{}_BLDG_FTPRINT_001.tif'.format(idx)

        AGL_img = rio.open(AGL_file)
        height = AGL_img.read(1)

        FACADE_img = rio.open(FACADE_file)
        facade = FACADE_img.read(1)

        roof_mask = bstool.generate_subclass_mask(facade, (6, 6,))
        masked_height = height * roof_mask
        masked_height[np.isnan(masked_height)] = 0.0

        height_99 = np.percentile(masked_height, 99.9999)
        height_01 = np.percentile(masked_height, 0.0001)

        masked_height[masked_height > height_99] = 0.0
        masked_height[masked_height < height_01] = 0.0

        BLDG_FTPRINT_img = rio.open(BLDG_FTPRINT_file)
        bldg_ftprint = BLDG_FTPRINT_img.read(1)
        footprint_mask = bstool.generate_subclass_mask(bldg_ftprint, (6, 6,))

        rgb = cv2.imread(RGB_file)

        vflow = mmcv.load(VFLOW_file)

        scale, angle = vflow['scale'], vflow['angle']

        offset_y = height * scale * np.cos(angle)
        offset_x = height * scale * np.sin(angle)

        offset_x, offset_y = offset_x.astype(np.int), offset_y.astype(np.int)

        XX, YY = np.meshgrid(np.arange(0, 2048), np.arange(0, 2048))
        offset_x += XX
        offset_y += YY

        offset_x = np.clip(offset_x, 0, 2047)
        offset_y = np.clip(offset_y, 0, 2047)

        roof_mask = footprint_mask[offset_y, offset_x]

        mask2polygons(rgb, masked_height, roof_mask)
