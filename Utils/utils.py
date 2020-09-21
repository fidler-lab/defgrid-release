import cv2
import numpy as np

def draw_poly(mask, poly):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly, dtype=np.int32)
    poly = poly.astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)

    return mask