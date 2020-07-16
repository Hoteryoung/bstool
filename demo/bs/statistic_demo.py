import bstool


if __name__ == '__main__':
    
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu', 'dalian_fine', 'xian_fine']
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    # cities = ['dalian_fine']
    csv_files = []
    title = []
    for city in cities:
        print("City: ", city)
        
        csv_file = f'./data/buildchange/v0/{city}/{city}_2048_footprint_gt.csv'
        csv_files.append(csv_file)
        title.append(city)

        statistic = bstool.Statistic(ann_file=None, csv_file=csv_files)
        
        # statistic.height_distribution(title)
        # statistic.height_curve(title)

        statistic.offset_distribution(title)
        statistic.offset_polar(title)
