import bstool
import pandas
import shapely
import os
import cv2
import csv


def parse_csv(csv_file):
    full_data = []
    csv_df = pandas.read_csv(csv_file)

    for index in csv_df.index:
        data = csv_df.loc[index].values[0:].tolist()
        data[0] = data[0].split('__')[0] + '__' + data[0].split('__')[1].split('_')[1] + '_' + data[0].split('__')[1].split('_')[0]
        full_data.append(data)

    return full_data

def bs_csv_dump(full_data, csv_file):
    with open(csv_file, 'w') as summary:
        csv_writer = csv.writer(summary, delimiter=',')
        csv_writer.writerow(['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])
        for data in full_data:
            csv_writer.writerow(data)

if __name__ == '__main__':

    src_csv_df = '/data/buildchange/public/20201028/xian_val_roof_crop1024_gt_minarea100.csv'

    dst_csv_df = '/data/buildchange/public/20201028/xian_val_roof_crop1024_gt_minarea100_fix.csv'

    full_data = parse_csv(src_csv_df)
    bs_csv_dump(full_data, dst_csv_df)




    