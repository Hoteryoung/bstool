import os
import bstool


if __name__ == "__main__":
    imagesets = ['val', 'train']
    folds = ['ATL']

    for imageset in imagesets:
        for fold in folds:
            print(f"========== processing {imageset} {fold} ==========")

            root_dir = f'./data/urban3d/v1/{fold}/{imageset}'
            json_dir = os.path.join(root_dir, 'weijia_labels')
            bstool.mkdir_or_exist(json_dir)
            
            csv_file = os.path.join(root_dir, '..', f'urban3d_{fold.lower()}_roof_offset_gt_simple_subcsv_merge_{imageset}.csv')
            csv_parser = bstool.CSVParse(csv_file, check_valid=False)
            image_name_list = csv_parser.image_name_list

            for image_name in image_name_list:
                # if image_name != 'JAX_Tile_163_RGB_001':
                #     continue
                # print(f'instance_gt_{imageset}/urban3d_{fold}_{imageset}_roof_offset_gt_subcsv_merge.csv')
                json_file = os.path.join(json_dir, f'{image_name}.json')
                image_info = {"ori_filename": f'{image_name}.tif',
                              "width": 1024,
                              "height": 1024}
                objects = csv_parser(image_name)
                if len(objects) == 0:
                    continue
                roof_polygons = []
                properties = []
                for obj in objects:
                    roof_polygon = obj['polygon']

                    data = dict()
                    data['offset'] = obj['offset']
                    data['building_height'] = obj['height']

                    roof_polygons.append(roof_polygon)
                    properties.append(data)

                # bstool.show_polygon(roof_polygons)

                bstool.urban3d_json_dump(roof_polygons, properties, image_info, json_file)


