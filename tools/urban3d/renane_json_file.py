import os


if __name__ == '__main__':
    versions = ['v1', 'v2']
    imagesets = ['val', 'train']
    folds = ['lables', 'images']
    
    root_dir = './data/urban3d'

    for version in versions:
        for imageset in imagesets:
            for fold in folds:
                dst_dir = os.path.join(root_dir, version, imageset, fold)
                for label_file in os.listdir(dst_dir):
                    if 'JSON' in label_file:
                        old_file = os.path.join(dst_dir, label_file)
                        new_file = os.path.join(dst_dir, label_file.replace('JSON', 'RGB'))
                        os.rename(old_file, new_file)