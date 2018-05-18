import cv2
import numpy as np

def image_loader(image_path):
    return cv2.imread(image_path)