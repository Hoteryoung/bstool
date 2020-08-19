import os

import bstool


if __name__ == '__main__':
    core_dataset_name = 'buildchange'

    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']

    src_version = 'v1'

    for city in cities:
            image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/images'
            edge_image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/edge_labels'

            edge_image_list = os.listdir(edge_image_dir)
            image_list = os.listdir(image_dir)

            for image_name in image_list:
                if image_name not in edge_image_list:
                    empty_edge_map = bstool.generate_image(1024, 1024, 0)
                    edge_file = os.path.join(edge_image_dir, image_name)
                    print(f"generate empty edge image: {edge_file}")
                    cv2.imwrite(edge_file, empty_edge_map)
            
