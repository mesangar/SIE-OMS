from skimage import feature
from scipy.ndimage import binary_dilation


def canny_images(img_arr):
    
    # Canny edge detector
    canny_img = feature.canny(img_arr, sigma=1)
    canny_imgi = canny_img * 255

    canny_imgi[canny_imgi < 230] = 0
    canny_imgi[canny_imgi > 230] = 255

    canny_imgi_dilated = binary_dilation(canny_imgi, structure=None, iterations=2)



    return canny_imgi_dilated

