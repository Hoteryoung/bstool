import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
import skimage
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
import json

import mmcv
import bstool


class Urban3D():
    def __init__(self,
                 src_dir,
                 dst_dir,
                 vis_dir,
                 camera_view,
                 show,
                 roof_move_mode='mean',
                 min_area=100):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.vis_dir = vis_dir
        bstool.mkdir_or_exist(self.dst_dir)
        self.camera_view = camera_view
        self.show = show
        self.roof_move_mode = roof_move_mode
        self.min_area = min_area

    def footprint2roof(self, footprint_polygon, offset):
        xoffset, yoffset = offset
        transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
        roof_polygon = affinity.affine_transform(footprint_polygon, transform_matrix)

        return roof_polygon
        
    def generate_v1(self):
        indexes = []
        for file_name in os.listdir(self.src_dir):
            if not file_name.endswith('.tif'):
                continue
            idx = file_name.split("_")[2]
            indexes.append(idx)
        indexes = list(set(indexes)) 

        for file_idx in [218]:
            save_json_file_name = f'JAX_Tile_{file_idx}_JSON_{self.camera_view}.json'
            print("Processing: ", self.camera_view, file_idx)
            AGL_file = os.path.join(src_dir, f'JAX_Tile_{file_idx}_AGL_{self.camera_view}.tif')
            FACADE_file = os.path.join(src_dir, f'JAX_Tile_{file_idx}_FACADE_{self.camera_view}.tif')
            RGB_file = os.path.join(src_dir, f'JAX_Tile_{file_idx}_RGB_{self.camera_view}.tif')
            VFLOW_file = os.path.join(src_dir, f'JAX_Tile_{file_idx}_VFLOW_{self.camera_view}.json')
            BLDG_FTPRINT_file = os.path.join(src_dir, f'JAX_Tile_{file_idx}_BLDG_FTPRINT_{self.camera_view}.tif')

            AGL_img = rio.open(AGL_file)
            height_mask = AGL_img.read(1)

            height_mask[np.isnan(height_mask)] = 0.0

            # height_99 = np.percentile(height_mask, 99.9999)
            # height_01 = np.percentile(height_mask, 0.0001)

            # height_mask[height_mask > height_99] = 0.0
            # height_mask[height_mask < height_01] = 0.0

            BLDG_FTPRINT_img = rio.open(BLDG_FTPRINT_file)
            bldg_ftprint = BLDG_FTPRINT_img.read(1)
            footprint_mask = bstool.generate_subclass_mask(bldg_ftprint, (6, 6,))

            rgb = cv2.imread(RGB_file)

            vflow = mmcv.load(VFLOW_file)

            scale, angle = vflow['scale'], vflow['angle']

            offset_y = height_mask * scale * np.cos(angle)
            offset_x = height_mask * scale * np.sin(angle)

            offset_y[np.isnan(offset_y)] = 0.0
            offset_x[np.isnan(offset_x)] = 0.0

            offset_x, offset_y = offset_x.astype(np.int), offset_y.astype(np.int)

            XX, YY = np.meshgrid(np.arange(0, rgb.shape[1]), np.arange(0, rgb.shape[0]))
            x_moved_coordinate = offset_x + XX
            y_moved_coordinate = offset_y + YY

            x_moved_coordinate = np.clip(x_moved_coordinate, 0, rgb.shape[1] - 1)
            y_moved_coordinate = np.clip(y_moved_coordinate, 0, rgb.shape[0] - 1)

            footprint_objects = bstool.mask_parse(footprint_mask)
            footprint_polygons = [obj['polygon'] for obj in footprint_objects]

            roof_mask = footprint_mask[y_moved_coordinate, x_moved_coordinate]

            roof_objects = bstool.mask_parse(roof_mask)
            roof_polygons = [obj['polygon'] for obj in roof_objects]

            annos = []
            if self.roof_move_mode == 'mean':
                roof_polygons = []
                for idx, footprint_polygon in enumerate(footprint_polygons):
                    object_structure = {}
                    foreground = bstool.generate_image(rgb.shape[0], rgb.shape[1], 0)

                    mask = bstool.polygon2mask(footprint_polygon)
                    mask = np.array(mask).reshape(1, -1, 2)
                    cv2.fillPoly(foreground, mask, 1)

                    polygon_inds = np.where(foreground != 0)

                    footprint_offset_x = offset_x[polygon_inds].mean()
                    footprint_offset_y = offset_y[polygon_inds].mean()

                    roof_polygon = self.footprint2roof(footprint_polygon, [footprint_offset_x, footprint_offset_y])
                    foreground = bstool.generate_image(rgb.shape[0], rgb.shape[1], 0)
                    mask = bstool.polygon2mask(roof_polygon)
                    mask = np.array(mask).reshape(1, -1, 2)
                    cv2.fillPoly(foreground, mask, 1)

                    roof_polygons.append(roof_polygon)

                    polygon_inds = np.where(foreground != 0)
                    polygon_height = height_mask[polygon_inds].mean()

                    object_structure['roof'] = bstool.polygon2mask(roof_polygon)
                    object_structure['footprint'] = bstool.polygon2mask(footprint_polygon)
                    object_structure['offset'] = [-float(footprint_offset_x), -float(footprint_offset_y)]
                    object_structure['building_height'] = float(polygon_height)

                    annos.append(object_structure)
            elif self.roof_move_mode == 'pixel':
                for idx, roof_polygon in enumerate(roof_polygons):
                    object_structure = {}
                    foreground = bstool.generate_image(rgb.shape[0], rgb.shape[1], 0)

                    mask = bstool.polygon2mask(roof_polygon)
                    mask = np.array(mask).reshape(1, -1, 2)
                    cv2.fillPoly(foreground, mask, 1)

                    polygon_inds = np.where(foreground != 0)

                    footprint_offset_x = offset_x[polygon_inds].mean()
                    footprint_offset_y = offset_y[polygon_inds].mean()

                    polygon_height = height_mask[polygon_inds].mean()

                    object_structure['roof'] = bstool.polygon2mask(roof_polygon)
                    # object_structure['footprint'] = bstool.polygon2mask(footprint_polygon)
                    object_structure['offset'] = [float(footprint_offset_x), float(footprint_offset_y)]
                    object_structure['building_height'] = float(polygon_height)

                    annos.append(object_structure)
            else:
                raise(RuntimeError(f'not support {self.roof_move_mode}'))

            if self.show:
                plt.imshow(rgb)
                plt.axis('off')

                for polygon in roof_polygons:
                    if type(polygon) == str:
                        polygon = shapely.wkt.loads(polygon)

                    plt.plot(*polygon.exterior.xy)

                plt.savefig(os.path.join(self.vis_dir, f'{file_idx}_{self.roof_move_mode}.png'), bbox_inches='tight', dpi=600, pad_inches=0.1)
                plt.show()

            image_info = {"ori_filename": f'JAX_Tile_{file_idx}_RGB_{self.camera_view}.tif',
                          "width": rgb.shape[1],
                          "height": rgb.shape[0],
                          "camera_view": self.camera_view}

            json_data = {"image": image_info,
                         "annotations": annos}

            json_file = os.path.join(dst_dir, save_json_file_name)
            with open(json_file, "w") as jsonfile:
                json.dump(json_data, jsonfile, indent=4)


if __name__ == '__main__':
    image_sets = ['JAX', 'OMA']

    for image_set in image_sets:
        processing_dir = f'./data/urban3d/v0/val/{image_set}_VAL/Val'
        
        camera_views = os.listdir(processing_dir)

        for camera_view in camera_views:
            src_dir = os.path.join(processing_dir, camera_view)
            dst_dir = './data/urban3d/v1/labels'
            vis_dir = './data/urban3d/vis'

            urban3d = Urban3D(src_dir,
                              dst_dir,
                              vis_dir,
                              camera_view,
                              show=True,
                              roof_move_mode='mean')

            urban3d.generate_v1()