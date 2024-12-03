import cv2
import numpy as np

def apply_thresholding(image, method="global"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "global":
        # Global thresholding
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif method == "adaptive":
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2)
    else:
        raise ValueError("Method must be either 'global' or 'adaptive'")

    return binary

    
