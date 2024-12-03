import cv2
import numpy as np

def apply_canny(image, lower_threshold=100, upper_threshold=200):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    # Convert edges to 3-channel format for visualization
    edges_colored = cv2.merge([edges, edges, edges])
    return edges_colored
