import phosphenes
from PIL import Image
import glob
import os
import scipy.misc
import numpy as np

path_list = glob.glob("/home/melani/Desktop/direct.png")

for path in sorted(path_list):
    img = Image.open(path).convert('L')
    img_array = np.asarray(img) / 255

    np.random.seed(10)

    img_phosphenes = phosphenes.convert_phosphenes(img_array, n_phosphenes_x=96, n_phosphenes_y=160, use_median=False, use_hex_pattern=False, dropout=0.1)

    # Save phosphene images results
    dir_path = os.path.split(path)
    filename = os.path.splitext(dir_path[1])
    #result_path = dir_path[0]  + '/phosp/' + filename[0] + filename[1]
    result_path = "/home/melani/Desktop/direct_phos.png"
    #print(result_path)
    scipy.misc.imsave(result_path, img_phosphenes)
