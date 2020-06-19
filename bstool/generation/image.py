import numpy as np
import cv2


def generate_image(height=512, 
                   width=512, 
                   color=(255, 255, 255)):
    if type(color) == tuple:
        b = np.full((height, width, 1), color[0], dtype=np.uint8)
        g = np.full((height, width, 1), color[1], dtype=np.uint8)
        r = np.full((height, width, 1), color[2], dtype=np.uint8)
        img = np.concatenate((b, g, r), axis=2)
    else:
        gray = np.full((height, width), color, dtype=np.uint8)
        img = gray

    return img

def generate_subclass_mask(mask_image,
                           subclasses=(1, 3)):
    height, width, _ = mask_image.shape
    sub_mask = generate_image(height, width, color=0)
    gray_mask_image = mask_image[:, :, 0]
    
    if isinstance(subclasses, (list, tuple)):
        keep_bool = np.logical_or(gray_mask_image == subclasses[0], gray_mask_image == subclasses[1])
    else:
        keep_bool = (gray_mask_image == subclasses)

    sub_mask[keep_bool] = 1

    return sub_mask                      