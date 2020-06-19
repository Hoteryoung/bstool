import PIL
import numpy as np
import cv2
import shapely
import matplotlib.pyplot as plt

import bstool


def show_polygons_on_image(masks, 
                           img, 
                           alpha=0.4, 
                           output_file=None):
    """show masks on image

    Args:
        masks (list): list of coordinate
        img (np.array): original image
        alpha (int): compress
        output_file (str): save path
    """
    color_list = list(bstool.COLORS.keys())
    img_h, img_w, _ = img.shape

    foreground = bstool.generate_image(img_h, img_w, (0, 0, 0))
    for idx, mask in enumerate(masks):
        mask = np.array(mask).reshape(1, -1, 2)
        cv2.fillPoly(foreground, mask, (bstool.COLORS[color_list[idx % 20]][2], bstool.COLORS[color_list[idx % 20]][1], bstool.COLORS[color_list[idx % 20]][0]))

    heatmap = bstool.show_grayscale_as_heatmap(foreground / 255.0, show=False, return_img=True)
    beta = (1.0 - alpha)
    fusion = cv2.addWeighted(heatmap, alpha, img, beta, 0.0)

    if output_file is not None:
        cv2.imwrite(output_file, fusion)
    else:
        bstool.show_image(fusion)

    return fusion

def show_polygon(polygons, size=[2048, 2048]):
    basic_size = 8
    plt.figure(figsize=(basic_size, basic_size * size[1] / float(size[0])))
    for polygon in polygons:
        if type(polygon) == str:
            polygon = shapely.wkt.loads(polygon)

        plt.plot(*polygon.exterior.xy)

    plt.xlim(0, size[0])
    plt.ylim(0, size[1])
    plt.gca().invert_yaxis()
    plt.show()