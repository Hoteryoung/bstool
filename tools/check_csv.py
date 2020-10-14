import bstool
import pandas
import shapely
import os
import cv2


if __name__ == '__main__':
    csv_file = '/data/urban3d/v1/ATL/urban3d_atl_roof_offset_gt_simple_subcsv_merge_val.csv'

    csv_parser = bstool.CSVParse(csv_file)

    for image_fn in csv_parser.image_fns:
        csv_parser(image_fn)