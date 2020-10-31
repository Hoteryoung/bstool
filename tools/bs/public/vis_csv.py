import bstool
import pandas
import shapely
import os
import cv2


if __name__ == '__main__':

    image_dir = '/data/buildchange/public/20201028/xian_fine/images'
    # csv_df = pandas.read_csv('/data/urban3d/weijia/urban3d_jax_oma_val_orgfootprint_offset_gt_simple_subcsv_merge.csv')
    # csv_df = pandas.read_csv('/data/buildchange/public/20201028/xian_val_footprint_crop1024_gt_minarea100_fix.csv')
    csv_df = pandas.read_csv('/data/buildchange/results/bc_v100.01.04_offset_rcnn_r50_2x_public_20201028_lr_0.02/bc_v100.01.04_offset_rcnn_r50_2x_public_20201028_lr_0.02_xian_public_footprint_splitted.csv')


    for image_name in os.listdir(image_dir):
        print(image_name)
        image_file = os.path.join(image_dir, image_name)
        image_basename = bstool.get_basename(image_name)

        img = cv2.imread(image_file)
 
        roof_masks = []
        for idx, row in csv_df[csv_df.ImageId == image_basename].iterrows():
            if type(row.PolygonWKT_Pix) == str:
                roof_polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
            else:
                roof_polygon = row.PolygonWKT_Pix

            roof_mask = bstool.polygon2mask(roof_polygon)
            roof_masks.append(roof_mask)

        if len(roof_masks) == 0:
            continue
        
        img = bstool.draw_masks_boundary(img, roof_masks)
        bstool.show_image(img)