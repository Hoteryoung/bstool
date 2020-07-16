import bstool


if __name__ == '__main__':
    
    # cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu', 'dalian_fine']
    cities = ['dalian_fine']
    for city in cities:
        csv_files = []
        title = []
        
        csv_file = f'./data/buildchange/v0/{city}/{city}_2048_footprint_gt.csv'
        csv_files.append(csv_file)
        title.append(city)

        statistic = bstool.Statistic(ann_file=None, csv_file=csv_files)
        
        # statistic.height_distribution(title)
        # statistic.height_curve(title)

        # statistic.offset_distribution(title)
        statistic.offset_polar(title)
