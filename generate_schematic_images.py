import schematic
from PIL import Image
import os
import numpy as np
import scipy.misc
import glob
from scipy import ndimage, misc
from scipy.misc import imresize

# Edges image path
Edges_path_list = glob.glob("your path")
# Object image path
Objects_path_list = glob.glob("your path")

for Edges_path, Objects_path in zip(sorted(Edges_path_list), sorted(Objects_path_list)):
    # Open images
    Edges_img = ndimage.imread(Edges_path)

    Objects_img = ndimage.imread(Objects_path, mode='L')
    Objects_img = imresize(Objects_img, (480, 480), interp='bilinear', mode=None)
    print(Edges_img.shape)
    print(Objects_img.shape)

    # Call function
    new_image = schematic.schematic_images(Edges_img, Objects_img)


    # save result image
    dir_path1 = os.path.split(Edges_path)
    filename1 = os.path.splitext(dir_path1[1])
    result_path1 = dir_path1[0] + "/SIE_OMS/" + 'SIE_OMS_' + filename1[0] + filename1[1]
    scipy.misc.imsave(result_path1, new_image)
