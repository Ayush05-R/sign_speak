"""
Shared image helpers.
"""
import cv2


def resize_for_detection(bgr_image, min_side: int = 0, max_side: int = 0):
    """
    Resize image for detection while preserving aspect ratio.
    - If min_side > 0, upscales so the shortest side >= min_side.
    - If max_side > 0, downscales so the longest side <= max_side.
    """
    if bgr_image is None:
        return bgr_image
    if min_side <= 0 and max_side <= 0:
        return bgr_image
    h, w = bgr_image.shape[:2]
    scale = 1.0
    if min_side > 0 and min(h, w) < min_side:
        scale = max(scale, float(min_side) / float(min(h, w)))
    if max_side > 0 and max(h, w) > max_side:
        scale = min(scale, float(max_side) / float(max(h, w)))
    if abs(scale - 1.0) < 1e-6:
        return bgr_image
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
