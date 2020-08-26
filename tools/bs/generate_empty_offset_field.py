import os
import cv2
import bstool


if __name__ == '__main__':
    core_dataset_name = 'buildchange'

    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']

    src_version = 'v1'

    counter = 0
    for city in cities:
        image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/images'
        offset_field_dir = f'./data/{core_dataset_name}/{src_version}/{city}/offset_field'

        offset_field_list = os.listdir(offset_field_dir)
        image_list = os.listdir(image_dir)

        for image_name in image_list:
            if bstool.get_basename(image_name) + '.npy' not in offset_field_list:
                empty_offset_field = bstool.generate_image(1024, 1024, 0)
                offset_field_file = os.path.join(offset_field_dir, image_name)
                counter += 1
                print(f"generate empty edge image: {offset_field_file}")
                # cv2.imwrite(offset_field_file, empty_offset_field)

    print("empty offset field", counter)
            
