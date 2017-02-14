import numpy as np
import cv2
import scipy.ndimage as sndi

def get_cropped_images(images):
    shape = images[0].shape
    height = shape[0]
    width = shape[1]
    return images[0:len(images), 65:height-35, 0:width]

def get_images_in_yuv(images):
    yuv_images = []
    for i in range(len(images)):
        yuv_images.append(cv2.cvtColor(images[i], cv2.COLOR_RGB2YUV))
    return np.array(yuv_images)

def get_resized_images(images):
    resized_images = []
    for i in range(len(images)):
        resized_images.append(cv2.resize(images[i].astype(np.uint8), (200, 66), cv2.INTER_AREA))
    return np.array(resized_images)

def get_blured_images(images):
    blured_images = []
    for i in range(len(images)):
        blured_images.append(cv2.bilateralFilter(images[i],9,75,75))
    return np.array(blured_images)

def preprocess_input(images):
    images = get_cropped_images(images)
    images = get_resized_images(images)
    images = get_images_in_yuv(images)
    images = get_blured_images(images)

    return images
