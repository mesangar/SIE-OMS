import downsampling
from PIL import Image
import os
import numpy as np
import scipy.misc
import glob

# Image path

path_list = glob.glob("your path")
for path in (path_list):

    img = Image.open(path)
    img = img.resize((480,480))
    # Call function
    gray_image = downsampling.gray_images(img)

    dir_path = os.path.split(path)
    filename = os.path.splitext(dir_path[1])
    result_path = dir_path[0] + "/Direct/" + 'Direct_' + filename[0] + filename[1]
    scipy.misc.imsave(result_path, gray_image)
