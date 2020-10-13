import os
import tqdm
import bstool


if __name__ == "__main__":
    imagesets = ['val', 'train']
    folds = ['ATL']

    for imageset in imagesets:
        for fold in folds:
            print(f"========== processing {imageset} {fold} ==========")

            root_dir = f'./data/urban3d/v1/{fold}/{imageset}'

            image_dir = os.path.join(root_dir, 'images')
            csv_dir = '/mnt/lustrenew/liweijia/data/urban_3d/instance_gt_atl/urban3d_atl_roof_offset_gt_simple_subcsv'
            json_dir = os.path.join(root_dir, 'weijia_labels')
            bstool.mkdir_or_exist(json_dir)
            
            for image_name in tqdm.tqdm(os.listdir(image_dir)):
                image_basename = bstool.get_basename(image_name)
                csv_file = os.path.join(csv_dir, image_basename + '.csv')
                csv_parser = bstool.CSVParse(csv_file, check_valid=False)

                json_file = os.path.join(json_dir, f'{image_basename}.json')
                image_info = {"ori_filename": f'{image_basename}.tif',
                            "width": 1024,
                            "height": 1024}
                objects = csv_parser(image_basename)
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


