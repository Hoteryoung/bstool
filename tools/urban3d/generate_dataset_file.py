import os


if __name__ == '__main__':
    versions = ['v1', 'v2']
    imagesets = ['train', 'val']
    datasets = ['JAX_OMA', 'ATL']
    
    root_dir = './data/urban3d'

    for version in versions:
        for imageset in imagesets:
            for dataset in datasets:
                image_dir = os.path.join(root_dir, version, imageset, 'images')
                save_file = os.path.join(root_dir, version, imageset, f'{dataset}_imageset_file.txt')
                with open(save_file, 'w+') as f:
                    for image_name in os.listdir(image_dir):
                        if image_name.split('_')[0] not in dataset:
                            continue
                        else:
                            f.write(image_name.split('.')[0] + '\n')