import cv2

import bstool

if __name__ == '__main__':
    image_file = '/data/plane/v0/train/images/1.tif'
    img = cv2.imread(image_file)

    img = bstool.image_translation(img, 200, -200)

    bstool.show_image(img)