import bstool


if __name__ == '__main__':
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    csv_files = []
    title = []
    for city in cities:
        print("City: ", city)
        
        csv_file = f'./data/buildchange/v0/{city}/{city}_2048_footprint_gt.csv'
        csv_files.append(csv_file)
        title.append(city)

    statistic = bstool.Statistic(ann_file=None, csv_file=csv_files)
    statistic.offset_polar(title)
    statistic.height_distribution(title)
