import bstool
import cv2


if __name__ == '__main__':
    img_file = '/data/buildchange/v0/shanghai/arg/final_edge5c_label/L18_106968_219320.png'
    img = cv2.imread(img_file)

    sub_mask = bstool.generate_subclass_mask(img, subclasses=(5, 6, 7, 8))

    print(sub_mask.max(), sub_mask.min())

    bstool.show_image(sub_mask * 255)
    # bstool.show_grayscale_as_heatmap(sub_mask)