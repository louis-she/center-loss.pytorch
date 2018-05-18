import cv2
import numpy as np

def image_loader(image_path):
    image = cv2.resize(cv2.imread(image_path), (96, 112))[:,:,::-1]
    return np.ascontiguousarray(image, dtype=np.float32)