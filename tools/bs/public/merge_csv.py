import bstool
import pandas
import shapely
import os
import cv2
import csv


def parse_csv(xian_csv_file, shanghai_csv_file):
    xian_full_data, shanghai_full_data = [], []

    xian_csv_df = pandas.read_csv(xian_csv_file)
    shanghai_csv_df = pandas.read_csv(shanghai_csv_file)

    for index in xian_csv_df.index:
        data = xian_csv_df.loc[index].values[0:].tolist()
        xian_full_data.append(data)

    for index in shanghai_csv_df.index:
        data = shanghai_csv_df.loc[index].values[0:].tolist()
        shanghai_full_data.append(data)

    return xian_full_data + shanghai_full_data

def bs_csv_dump(full_data, csv_file):
    with open(csv_file, 'w') as summary:
        csv_writer = csv.writer(summary, delimiter=',')
        csv_writer.writerow(['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])
        for data in full_data:
            # if type(data[2]) == str:
            #     polygon = shapely.wkt.loads(data[2])
            # else:
            #     polygon = data[2]
            # if not bstool.single_valid_polygon(polygon):
            #     data[2] = polygon.buffer(0).wkt
            #     # continue
            csv_writer.writerow(data)

if __name__ == '__main__':
    xian_csv_format = '/data/buildchange/public/20201028/xian_val_{}_crop1024_gt_minarea500.csv'
    shanghai_csv_format = '/data/buildchange/public/20201028/shanghai_val_v3_final_crop1024_{}_gt_minarea500.csv'

    merge_csv_format = '/data/buildchange/public/20201028/shanghai_xian_v3_merge_val_{}_crop1024_gt_minarea500.csv'
    
    for mask in ['roof', 'footprint']:
        xian_csv_file = xian_csv_format.format(mask)
        shanghai_csv_file = shanghai_csv_format.format(mask)

        merge_csv_file = merge_csv_format.format(mask)

        full_data = parse_csv(xian_csv_file, shanghai_csv_file)

        bs_csv_dump(full_data, merge_csv_file)




    