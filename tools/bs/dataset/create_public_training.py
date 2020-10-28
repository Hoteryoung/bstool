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
                   cities=['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu'], 
                   score_threshold_index=10000,
                   keep_sub_image_num_threshold=4):
    training_info = []
    sub_image_num = 0
    for city in cities:
        for candidate_coord in candidate_coords:
            arg = f"{city}_arg__{ori_image_name}__{candidate_coord[0]}_{candidate_coord[1]}"
            google = f"{city}_google__{ori_image_name}__{candidate_coord[0]}_{candidate_coord[1]}"
            ms = f"{city}_ms__{ori_image_name}__{candidate_coord[0]}_{candidate_coord[1]}"

            if arg in file_names and google in file_names and ms in file_names:
                try:
                    arg_idx, google_idx, ms_idx = file_names.index(arg, 0, score_threshold_index), file_names.index(google, 0, score_threshold_index), file_names.index(ms, 0, score_threshold_index)
                except:
                    continue
                
                sub_image_num += 1
    
                training_info.append(full_data[arg_idx])
                training_info.append(full_data[google_idx])
                training_info.append(full_data[ms_idx])

    # print("sub_image_num: ", sub_image_num)
    if sub_image_num >= keep_sub_image_num_threshold or keep_sub_image_num_threshold == 0:
        return training_info
    else:
        return

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval on semantic segmentation')
    parser.add_argument(
        '--score_threshold',
        type=int,
        default=13608, # 17225 for 20201028 generation, 13608 for 20201027 generation
        help='dataset for evaluation')
    parser.add_argument(
        '--keep_threshold',
        type=int,
        default=3, 
        help='dataset for evaluation')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    version = '20201027'
    csv_file = f'./data/buildchange/public/misc/nooverlap/full_dataset_info_{version}.csv'
    candidate_coords = [(0, 0), (0, 1024), (1024, 0), (1024, 1024)]
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    
    file_names, ori_image_names, scores, full_data = parse_csv(csv_file)

    score_threshold_index = args.score_threshold
    keep_sub_image_num_threshold = args.keep_threshold

    print("You set the thresholds of score and keep as: ", score_threshold_index, keep_sub_image_num_threshold)
    
    training_info = []
    for ori_image_name, score in zip(ori_image_names, scores):
        result = keep_ori_image(ori_image_name, 
                                file_names, 
                                candidate_coords, 
                                cities, 
                                score_threshold_index=score_threshold_index, 
                                keep_sub_image_num_threshold=keep_sub_image_num_threshold)
        if result is None:
            continue
        else:
            training_info.extend(result)
    
    training_csv_file = f'./data/buildchange/public/misc/nooverlap/training_dataset_info_{version}.csv'
    training_imageset_file = f'./data/buildchange/public/misc/nooverlap/training_imageset_file_{version}.txt'
    with open(training_csv_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        head = ['file_name', 'sub_fold', 'ori_image_fn', 'coord_x', 'coord_y', 'object_num', 'mean_angle', 'mean_height', 'mean_offset_length', 'std_offset_length', 'std_angle', 'no_ignore_rate', 'score']
        csv_writer.writerow(head)
        for data in training_info:
            csv_writer.writerow(data)

    print("The number of training data: ", len(training_info))

    f = open(training_imageset_file, 'w')
    if len(training_info) == 3000:
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

            f.write(f"{city} {sub_fold} {ori_image_name}\n")

    f.close()