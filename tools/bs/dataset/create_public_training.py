import os
import csv
import numpy as np
from collections import defaultdict
import pandas
import tqdm
import shutil

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

if __name__ == '__main__':
    csv_file = './data/buildchange/public/misc/nooverlap/full_dataset_info.csv'
    candidate_coords = [(0, 0), (0, 1024), (1024, 0), (1024, 1024)]
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']

    file_names, ori_image_names, scores, full_data = parse_csv(csv_file)

    score_threshold_index = 9360
    
    training_info = []
    for ori_image_name, score in zip(ori_image_names, scores):
        for candidate_coord in candidate_coords:
            for city in cities:
                arg = f"{city}_arg__{ori_image_name}__{candidate_coord[0]}_{candidate_coord[1]}"
                google = f"{city}_google__{ori_image_name}__{candidate_coord[0]}_{candidate_coord[1]}"
                ms = f"{city}_ms__{ori_image_name}__{candidate_coord[0]}_{candidate_coord[1]}"

                if arg in file_names and google in file_names and ms in file_names:
                    try:
                        arg_idx, google_idx, ms_idx = file_names.index(arg, 0, score_threshold_index), file_names.index(google, 0, score_threshold_index), file_names.index(ms, 0, score_threshold_index)
                    except:
                        continue
                
                    training_info.append(full_data[arg_idx])
                    training_info.append(full_data[google_idx])
                    training_info.append(full_data[ms_idx])

    
    training_csv_file = './data/buildchange/public/misc/nooverlap/training_dataset_info_20201027.csv'
    with open(training_csv_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        head = ['file_name', 'sub_fold', 'ori_image_fn', 'coord_x', 'coord_y', 'object_num', 'mean_angle', 'mean_height', 'mean_offset_length', 'std_offset_length', 'std_angle', 'no_ignore_rate', 'score']
        csv_writer.writerow(head)
        for data in training_info:
            csv_writer.writerow(data)

    print("The number of training data: ", len(training_info))
    if len(training_info) == 3000:
        src_root_image_dir = './data/buildchange/v2/{}/images'
        src_root_label_dir = './data/buildchange/v2/{}/labels'
        dst_root_image_dir = './data/buildchange/public/20201027/{}/images'
        dst_root_label_dir = './data/buildchange/public/20201027/{}/labels'
        for data in tqdm.tqdm(training_info):
            base_name = data[0]
            city = base_name.split("__")[0].split('_')[0]
            src_image_file = os.path.join(src_root_image_dir.format(city), base_name + '.png')
            src_label_file = os.path.join(src_root_label_dir.format(city), base_name + '.json')

            bstool.mkdir_or_exist(dst_root_image_dir.format(city))
            bstool.mkdir_or_exist(dst_root_label_dir.format(city))

            dst_image_file = os.path.join(dst_root_image_dir.format(city), base_name + '.png')
            dst_label_file = os.path.join(dst_root_label_dir.format(city), base_name + '.json')
            
            shutil.copy(src_image_file, dst_image_file)
            shutil.copy(src_label_file, dst_label_file)