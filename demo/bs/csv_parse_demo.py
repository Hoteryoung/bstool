import bstool


if __name__ == '__main__':
    csv_file = '/data/buildchange/v0/xian_fine/xian_val_footprint_gt_minarea100_26.csv'

    csv_parser =  bstool.CSVParse(csv_file, 100)

    polygons = []
    for image_name in csv_parser.image_name_list:
        objects = csv_parser(image_name)

        polygons += [obj['polygon'] for obj in objects]

    print(len(polygons))