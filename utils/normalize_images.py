import numpy as np

def normalize_images(images):
    images = images.reshape(images.shape[0], images.shape[1]).T
    images = images.astype('float32')
    images /= 255
    return images
