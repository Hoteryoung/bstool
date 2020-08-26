import os
import cv2
import numpy as np
import bstool


if __name__ == '__main__':
    core_dataset_name = 'buildchange'

    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']

    src_version = 'v1'

    fn_list = []

    counter = 0
    for city in cities:
        image_dir = f'./data/{core_dataset_name}/{src_version}/{city}/images'
        offset_field_dir = f'./data/{core_dataset_name}/{src_version}/{city}/offset_field'

        offset_field_list = os.listdir(offset_field_dir)
        image_list = os.listdir(image_dir)

        for image_name in image_list:
            if bstool.get_basename(image_name) + '.npy' not in offset_field_list:
                empty_offset_field = bstool.generate_image(1024, 1024, (0, 0, 0))
                offset_field_file = os.path.join(offset_field_dir, bstool.get_basename(image_name) + '.npy' )
                counter += 1
                
                original_fn = bstool.get_info_splitted_imagename(image_name)[1]
                fn_list.append(original_fn)
                print(f"generate empty edge image: {offset_field_file}")

                np.save(offset_field_file, empty_offset_field)
                
                # cv2.imwrite(offset_field_file, empty_offset_field)

    print("empty offset field", counter)
    print("empty offset field: ", set(fn_list))
            
