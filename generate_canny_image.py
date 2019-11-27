import canny
from PIL import Image
import os
import numpy as np
import scipy.misc
import glob

# Image path

path_list = glob.glob("/home/melani/Desktop/vv/Office_1.png")
for path in (path_list):

    img_g = Image.open(path).convert('L')
    img_g = img_g.resize((480,480))
    img_arr = np.asarray(img_g)

    # Call function
    canny_imgi_dilated = canny.canny_images(img_arr)

    dir_path = os.path.split(path)
    filename = os.path.splitext(dir_path[1])
    result_path = dir_path[0] + '/Canny_' + filename[0] + filename[1]
    scipy.misc.imsave(result_path, canny_imgi_dilated)