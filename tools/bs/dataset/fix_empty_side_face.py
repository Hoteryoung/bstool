import os
import cv2
import bstool


if __name__ == '__main__':
    core_dataset_name = 'buildchange'

    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']

    src_version = 'v1'

    for city in cities:
        image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/images'
        side_face_image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/side_face'

        side_face_image_list = os.listdir(side_face_image_dir)
        image_list = os.listdir(image_dir)

        counter = 0
        for image_name in side_face_image_list:
            # empty_edge_map = bstool.generate_image(1024, 1024, 0)
            
            side_face_file = os.path.join(side_face_image_dir, image_name)
            side_face = cv2.imread(side_face_file)

            if side_face.sum() == 0:
                counter += 1
                print(f"{counter} generate empty side face image: {side_face_file}")
            
            # cv2.imwrite(side_face_file, empty_edge_map)
            
