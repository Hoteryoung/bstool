import os


if __name__ == '__main__':
    versions = ['v2', 'v1']
    imagesets = ['val', 'train']
    datasets = ['JAX_OMA', 'ATL']
    
    root_dir = './data/urban3d'

    for version in versions:
        for imageset in imagesets:
            for dataset in datasets:
                image_dir = os.path.join(root_dir, version, imageset, 'images')
                save_file = os.path.join(root_dir, version, imageset, f'{dataset}_imageset_file.txt')
                if os.path.exists(save_file):
                    os.remove(save_file)
                with open(save_file, 'w+') as f:
                    for image_name in os.listdir(image_dir):
                        if image_name.split('_')[0] not in dataset:
                            continue
                        else:
                            f.write(image_name.split('.')[0] + '\n')