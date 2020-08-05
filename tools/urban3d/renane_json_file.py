import os


if __name__ == '__main__':
    versions = ['v1', 'v2']
    imagesets = ['val', 'train']
    
    root_dir = './data/urban3d'

    for version in versions:
        for imageset in imagesets:
            label_dir = os.path.join(root_dir, version, imageset, 'labels')
            for label_file in os.listdir(label_dir):
                if 'JSON' in label_file:
                    old_file = os.path.join(label_dir, label_file)
                    new_file = os.path.join(label_dir, label_file.replace('JSON', 'RGB'))
                    os.rename(old_file, new_file)