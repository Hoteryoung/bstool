import bstool
import pandas
import shapely
import os
import cv2


if __name__ == '__main__':
    model = 'bc_v015_mask_rcnn_r50_v2_roof_trainval'

    image_set = 'dalian_fine'

    anno_file = f'./data/buildchange/v1/coco/annotations/buildchange_v1_val_{image_set}.json'
    pkl_file = f'/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/{model}/coco_results_{image_set}_v015.pkl'
    csv_prefix = f'/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/{model}/{model}_{image_set}'
    image_dir = f'/data/buildchange/v1/{image_set}/images'

    bstool.pkl2csv_roof_footprint(pkl_file, anno_file, csv_prefix, score_threshold=0.05)

    roof_df = pandas.read_csv(csv_prefix + "_" + 'roof' + '.csv')

    for image_name in os.listdir(image_dir):
        image_file = os.path.join(image_dir, image_name)
        image_basename = bstool.get_basename(image_name)

        img = cv2.imread(image_file)

        roof_masks = []
        for idx, row in roof_df[roof_df.ImageId == image_basename].iterrows():
            roof_polygon = shapely.wkt.loads(row.PolygonWKT_Pix)
            if not roof_polygon.is_valid:
                continue
            roof_mask = bstool.polygon2mask(roof_polygon)
            roof_masks.append(roof_mask)
        
        img = bstool.draw_masks_boundary(img, roof_masks)
        bstool.show_image(img)