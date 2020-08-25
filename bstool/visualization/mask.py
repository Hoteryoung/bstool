import numpy as np
import cv2
import shapely
import matplotlib.pyplot as plt

import bstool


def show_masks_on_image(img,
                        masks,
                        alpha=0.4,
                        show=True,
                        output_file=None,
                        win_name=''):
    """show masks on image

    Args:
        img (np.array): original image
        masks (list): list of masks, mask = [x1, y1, x2, y2, ....]
        alpha (int): compress
        show (bool): show flag
        output_file (str): save path
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    img_h, img_w, _ = img.shape

    color_list = list(bstool.COLORS.keys())

    foreground = bstool.generate_image(img_h, img_w, (0, 0, 0))
    for idx, mask in enumerate(masks):
        mask = np.array(mask).reshape(1, -1, 2)
        cv2.fillPoly(foreground, mask, (bstool.COLORS[color_list[idx % 20]][2], bstool.COLORS[color_list[idx % 20]][1], bstool.COLORS[color_list[idx % 20]][0]))

    heatmap = bstool.show_grayscale_as_heatmap(foreground / 255.0, show=False, return_img=True)
    beta = (1.0 - alpha)
    fusion = cv2.addWeighted(heatmap, alpha, img, beta, 0.0)

    if show:
        bstool.show_image(fusion, output_file=output_file, win_name=win_name)

    return fusion

def show_polygons_on_image(img,
                           polygons,
                           alpha=0.4,
                           show=True,
                           output_file=None,
                           win_name=''):
    """show polygons on image

    Args:
        img (np.array): original image
        polygons (list): list of polygons
        alpha (int): compress
        show (bool): show flag
        output_file (str): save path
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    img_h, img_w, _ = img.shape

    color_list = list(bstool.COLORS.keys())

    foreground = bstool.generate_image(img_h, img_w, (0, 0, 0))
    for idx, polygon in enumerate(polygons):
        mask = bstool.polygon2mask(polygon)
        mask = np.array(mask).reshape(1, -1, 2)
        cv2.fillPoly(foreground, mask, (bstool.COLORS[color_list[idx % 20]][2], bstool.COLORS[color_list[idx % 20]][1], bstool.COLORS[color_list[idx % 20]][0]))

    heatmap = bstool.show_grayscale_as_heatmap(foreground / 255.0, show=False, return_img=True)
    beta = (1.0 - alpha)
    fusion = cv2.addWeighted(heatmap, alpha, img, beta, 0.0)

    if show:
        bstool.show_image(fusion, output_file=output_file, win_name=win_name)

    return fusion

def show_polygon(polygons, size=[2048, 2048], output_file=None):
    """show polygons

    Args:
        polygons (list): list of polygon
        size (list, optional): image size . Defaults to [2048, 2048].
    """
    basic_size = 8
    plt.figure(figsize=(basic_size, basic_size * size[1] / float(size[0])))
    for polygon in polygons:
        if type(polygon) == str:
            polygon = shapely.wkt.loads(polygon)

        plt.plot(*polygon.exterior.xy)

    plt.xlim(0, size[0])
    plt.ylim(0, size[1])
    plt.gca().invert_yaxis()
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight', dpi=600, pad_inches=0.0)
    plt.show()

def show_coco_mask(coco, image_file, anns, output_file=None):
    img = cv2.imread(image_file)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    
    coco.showAnns(anns)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=600, pad_inches=0.0)
        plt.clf()
    else:
        plt.show()

def draw_masks_boundary(img, masks, color=(0, 0, 255), thickness=2):
    for mask in masks:
        img = draw_mask_boundary(img, mask, color=color, thickness=thickness)

    return img

def draw_mask_boundary(img, mask, color=(0, 0, 255), thickness=2):
    mask = np.array(mask).reshape((-1, 1, 2))
    img = cv2.polylines(img, [mask], True, color, thickness=thickness, lineType=cv2.LINE_AA)

    return img

def draw_iou(img, polygon, iou, color=(0, 0, 255)):
    centroid = bstool.get_polygon_centroid([polygon])[0]

    centroid = tuple([int(_) for _ in centroid])

    img = cv2.putText(img, "{:.2f}".format(iou), centroid, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = color, thickness = 2, lineType = 8)

    return img

def draw_height_angle(img, angle, color=(0, 0, 255)):
    img_height, img_width = img.shape[0], img.shape[1]

    coord = (int(img_height // 2), int(img_width // 2))

    img = cv2.putText(img, "{:.2f}".format(angle * 180.0 / np.pi), coord, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = color, thickness = 2, lineType = 8)

    return img