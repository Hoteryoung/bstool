import os
import pandas
import csv
from collections import defaultdict

import bstool


if __name__ == '__main__':
    root_dir = '/mnt/lustre/wangjinwang/data/urban3d/v1/train/ATL'
    src_csv_files = ['urban3d_atl_orgfootprint_offset_gt_simple_subcsv_merge.csv', 
                     'urban3d_atl_roof_offset_gt_simple_subcsv_merge.csv']
    
    imageset_names_dict = defaultdict(list)
    for imageset in ['train', 'val']:
        imageset_file = f'/mnt/lustrenew/liweijia/data/urban_3d/ATL/list_trainval/region-split-v1/atl_imageid_region_{imageset}.txt'
        with open(imageset_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            imageset_names_dict[imageset].append(line.strip('\n'))

    for src_csv_file in src_csv_files:
        src_csv_file = os.path.join(root_dir, src_csv_file)
        for imageset in ['train', 'val']:
            dst_csv_file = os.path.join(root_dir, bstool.get_basename(src_csv_file) + f'_{imageset}.csv')
            imageset_names = imageset_names_dict[imageset]

            with open(src_csv_file, 'r', newline='') as f:
                csv_reader = csv.reader(f)

                save_items = []
                for idx, row in enumerate(csv_reader):
                    if idx == 0 or row[0] in imageset_names:
                        save_items.append(row)

            print(f"{src_csv_file} {imageset} {dst_csv_file} item length: ", len(save_items) - 1)
            with open(dst_csv_file, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                for item in save_items:
                    csv_writer.writerow(row)
