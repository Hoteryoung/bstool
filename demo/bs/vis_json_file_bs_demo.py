import os
import cv2

import bstool


if __name__ == '__main__':
    image_dir = './data/buildchange/v1/xian_fine/images'
    label_dir = './data/buildchange/v1/xian_fine/labels'
    
    vis_dir = './data/buildchange/v1/xian_fine/vis'
    bstool.mkdir_or_exist(vis_dir)

    for image_name in os.listdir(image_dir):
        print(image_name)
        file_name = bstool.get_basename(image_name)
        rgb_file = os.path.join(image_dir, image_name)
        json_file = os.path.join(label_dir, file_name + '.json')

        objects = bstool.urban3d_json_parse(json_file)

        if len(objects) == 0:
            continue

        footprint_masks = [obj['footprint_mask'] for obj in objects]
        footprint_polygons = [bstool.mask2polygon(obj['footprint_mask']) for obj in objects]

        bboxes = [obj['footprint_bbox'] for obj in objects]
        bboxes = [bstool.xywh2xyxy(bbox) for bbox in bboxes]
        # bstool.show_bboxs_on_image(rgb_file, bboxes, win_name='footprint bbox')

        img = cv2.imread(rgb_file)

        img_show = bstool.draw_masks_boundary(img.copy(), footprint_masks)
        # bstool.show_image(img_show)
        # fusion = bstool.show_masks_on_image(rgb_file, footprint_masks, win_name='footprint mask', show=False)
        cv2.imwrite(os.path.join(vis_dir, file_name + '_footprint.png'), img_show)


        offsets = [obj['offset'] for obj in objects]
        roof_polygons = [bstool.roof2footprint_single(footprint_polygon, offset, 'roof2footprint') for footprint_polygon, offset in zip(footprint_polygons, offsets)]
        roof_masks = [bstool.polygon2mask(roof_polygon) for roof_polygon in roof_polygons]
        
        img_show = bstool.draw_masks_boundary(img.copy(), roof_masks)
        # bstool.show_image(img_show)
        # masks = [obj['roof_mask'] for obj in objects]
        # fusion = bstool.show_masks_on_image(rgb_file, roof_masks, win_name='roof mask', show=False)
        cv2.imwrite(os.path.join(vis_dir, file_name + '_roof.png'), img_show)