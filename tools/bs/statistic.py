import bstool


if __name__ == '__main__':
    ann_file = '/data/buildchange/v1/coco/annotations/buildchange_v1_train_shanghai.json'
    
    # cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    cities = ['dalian_fine']
#  'xian_fine', 'dalian_fine'
    csv_files = []
    title = []
    for city in cities:
        csv_file = f'./data/buildchange/v0/{city}/{city}_2048_footprint_gt.csv'
        csv_files.append(csv_file)
        title.append(city)

    statistic = bstool.Statistic(ann_file=None, csv_file=csv_files)
    
    statistic.height(title)