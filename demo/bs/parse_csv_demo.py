import os
import cv2

import bstool


if __name__ == '__main__':
    gt_footprint_csv_file = '/data/buildchange/v0/dalian_fine/dalian_fine_2048_footprint_gt.csv'

    csv_parser = bstool.CSVParse(gt_footprint_csv_file)

    for image_name in csv_parser.image_name_list:
        objects = csv_parser(image_name)

        print(objects)