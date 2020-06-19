import cv2
import numpy as np

import bstool


def show_grayscale_as_heatmap(grayscale_image, 
                              show=True,
                              win_name='',
                              wait_time=0,
                              return_img=False):
    """show grayscale image as rgb image

    Args:
        grayscale_image (np.array): gray image
        show (bool, optional): show flag. Defaults to True.
        win_name (str, optional): windows name. Defaults to ''.
        wait_time (int, optional): wait time. Defaults to 0.
        return_img (bool, optional): return colored image. Defaults to False.

    Returns:
        np.array: colored image
    """
    grayscale_image = grayscale_image.astype(np.float64)
    max_value = np.max(grayscale_image)
    min_value = np.min(grayscale_image)
    grayscale_image = 255 * (grayscale_image - min_value) / (max_value - min_value)
    grayscale_image = grayscale_image.astype(np.uint8)
    heatmap_image = cv2.applyColorMap(grayscale_image, cv2.COLORMAP_JET)

    if show:
        cv2.imshow(win_name, heatmap_image)
        cv2.waitKey(wait_time)
    
    if return_img:
        return heatmap_image

def show_image(img, 
               output_file=None,
               win_name='',
               win_size=800,
               wait_time=0):
    """show image

    Args:
        img (np.array): input image
        win_name (str, optional): windows name. Defaults to ''.
        win_size (int, optional): windows size. Defaults to 800.
        wait_time (int, optional): wait time . Defaults to 0.
        output_file ([type], optional): save the image. Defaults to None.

    """
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_size, win_size)
    cv2.imshow(win_name, img)
    cv2.waitKey(wait_time)
    if output_file != None:
        dir_name = bstool.get_dir_name(output_file)
        bstool.mkdir_or_exist(dir_name)

        cv2.imwrite(output_file, img)