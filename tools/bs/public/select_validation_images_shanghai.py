import os
import csv
import numpy as np
from collections import defaultdict
import pandas
import tqdm
import shutil
import argparse

import bstool


def parse_csv(csv_file):
    csv_df = pandas.read_csv(csv_file)
    file_names = list(csv_df.file_name.unique())
    ori_image_names = list(csv_df.ori_image_fn.unique())
    scores = list(csv_df.score.unique())

    full_data = []

    for index in csv_df.index:
        full_data.append(csv_df.loc[index].values[0:])

    return file_names, ori_image_names, scores, full_data

def keep_ori_image(ori_image_name, 
                   file_names, 
                   candidate_coords=[(0, 0), (0, 1024), (1024, 0), (1024, 1024)], 
                   sub_folds=['arg', 'google', 'ms'], 
                   score_threshold_index=10000,
                   keep_sub_image_num_threshold=4):
    training_info = []
    sub_image_num = 0
    for sub_fold in sub_folds:
        sub_image1 = f"shanghai_{sub_fold}__{ori_image_name}__{candidate_coords[0][0]}_{candidate_coords[0][1]}"
        sub_image2 = f"shanghai_{sub_fold}__{ori_image_name}__{candidate_coords[1][0]}_{candidate_coords[1][1]}"
        sub_image3 = f"shanghai_{sub_fold}__{ori_image_name}__{candidate_coords[2][0]}_{candidate_coords[2][1]}"
        sub_image4 = f"shanghai_{sub_fold}__{ori_image_name}__{candidate_coords[3][0]}_{candidate_coords[3][1]}"

        if sub_image1 in file_names and sub_image2 in file_names and sub_image3 in file_names and sub_image4 in file_names:
            try:
                idx1, idx2, idx3, idx4 = file_names.index(sub_image1, 0, score_threshold_index), file_names.index(sub_image2, 0, score_threshold_index), file_names.index(sub_image3, 0, score_threshold_index), file_names.index(sub_image4, 0, score_threshold_index)
            except:
                continue
            

            training_info.append(full_data[idx1])
            training_info.append(full_data[idx2])
            training_info.append(full_data[idx3])
            training_info.append(full_data[idx4])

    return training_info

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval on semantic segmentation')
    parser.add_argument(
        '--score_threshold',
        type=int,
        default=17225, # 17225 for 20201028 generation, 13608 for 20201027 generation, 9360 and 0 for 20201027 highset generation
        help='dataset for evaluation')
    parser.add_argument(
        '--keep_threshold',
        type=int,
        default=4, 
        help='dataset for evaluation')
    parser.add_argument(
        '--version',
        type=str,
        default='20201028', 
        help='dataset for evaluation')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    version = args.version
    print("Processing the version of ", version)
    csv_file = f'./data/buildchange/public/misc/nooverlap/full_dataset_info_20201028_shanghai.csv'
    candidate_coords = [(0, 0), (0, 1024), (1024, 0), (1024, 1024)]
    sub_folds = ['arg', 'google', 'ms']
    
    file_names, ori_image_names, scores, full_data = parse_csv(csv_file)

    score_threshold_index = args.score_threshold
    keep_sub_image_num_threshold = args.keep_threshold

    print("You set the thresholds of score and keep as: ", score_threshold_index, keep_sub_image_num_threshold)
    
    training_info = []
    for ori_image_name in ori_image_names:
        result = keep_ori_image(ori_image_name, 
                                file_names, 
                                candidate_coords, 
                                sub_folds, 
                                score_threshold_index=score_threshold_index, 
                                keep_sub_image_num_threshold=keep_sub_image_num_threshold)
        if result is None:
            continue
        else:
            training_info.extend(result)
    
    training_csv_file = f'./data/buildchange/public/misc/nooverlap/validation_dataset_info_{version}_shanghai.csv'
    training_imageset_file = f'./data/buildchange/public/misc/nooverlap/validation_imageset_file_{version}_shanghai.txt'
    with open(training_csv_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        head = ['file_name', 'sub_fold', 'ori_image_fn', 'coord_x', 'coord_y', 'object_num', 'mean_angle', 'mean_height', 'mean_offset_length', 'std_offset_length', 'std_angle', 'no_ignore_rate', 'score']
        csv_writer.writerow(head)
        for data in training_info:
            csv_writer.writerow(data)

    print("The number of training data: ", len(training_info))

    selected_image_info = []
    f = open(training_imageset_file, 'w')
    if len(training_info) == 200:
        src_root_image_dir = './data/buildchange/v2/{}/images'
        src_root_label_dir = './data/buildchange/v2/{}/labels'
        dst_root_image_dir = './data/buildchange/public/{}/{}/images'
        dst_root_label_dir = './data/buildchange/public/{}/{}/labels'
        for data in tqdm.tqdm(training_info):
            base_name = data[0]
            city = base_name.split("__")[0].split('_')[0]
            sub_fold, ori_image_name, coord = bstool.get_info_splitted_imagename(base_name)
            src_image_file = os.path.join(src_root_image_dir.format(city), base_name + '.png')
            src_label_file = os.path.join(src_root_label_dir.format(city), base_name + '.json')

            bstool.mkdir_or_exist(dst_root_image_dir.format(version, city))
            bstool.mkdir_or_exist(dst_root_label_dir.format(version, city))

            dst_image_file = os.path.join(dst_root_image_dir.format(version, city), base_name + '.png')
            dst_label_file = os.path.join(dst_root_label_dir.format(version, city), base_name + '.json')
            
            shutil.copy(src_image_file, dst_image_file)
            shutil.copy(src_label_file, dst_label_file)

            info = f"{city} {sub_fold} {ori_image_name}\n"
            if info not in selected_image_info:
                f.write(info)
                selected_image_info.append(info)

    f.close()