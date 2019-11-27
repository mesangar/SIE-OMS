import numpy as np

def gray_images(img):
    # Canny edge detector
    gray_image = img.convert('L')

    gray_image = np.asarray(gray_image)

    return gray_image

