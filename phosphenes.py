import numpy as np


def generate_phosphenes_look_up (
        size_x,
        size_y,
        gray_levels = 8,
        sigma = 2):

    phosphenes = np.zeros((size_x, size_y, gray_levels))
    pos_x = size_x / 2.0
    pos_y = size_y / 2.0
    max_gray_levels = gray_levels
    #max_gray_levels = gray_levels -1

    x, y = np.meshgrid(range(0,size_x), range(0,size_y), indexing='ij')

    localDist2 = np.square(x - pos_x) + np.square(y - pos_y)
    for i in range(1,gray_levels):
        #phosphenes[:, :, i] = (np.exp(-localDist2 / (2 * np.square(sigma * i / max_gray_levels)))) * i / max_gray_levels
        phosphenes[:, :, i] = (np.exp(-localDist2 / (2 * np.square(sigma * (i+1) / max_gray_levels)))) * (i+1) / max_gray_levels
    return phosphenes



def convert_phosphenes ( img_array,
                         n_phosphenes_x=32,
                         n_phosphenes_y=32,
                         use_median = False,
                         use_hex_pattern = True,
                         dropout = 0.1):

    img_phosphenes = np.zeros(img_array.shape)
    width, height = img_array.shape

    stepx = width  // n_phosphenes_x
    stepy = height // n_phosphenes_y

    phosphenes_look_up = generate_phosphenes_look_up(stepx, stepy)
    gray_levels = phosphenes_look_up.shape[2]
    max_gray_levels = gray_levels - 1

    if stepx != stepy:
        print('WARNING: Phosphenes are not square.')

    even_row = True
    for start_i in range(0, width+stepx, stepx):
        even_row = not even_row
        end_i = start_i + stepx
        if end_i > width:
            end_i = width

        for start_j in range(0, height+stepy, stepy):
            offset = stepy // 2 if use_hex_pattern and even_row else 0
            start_j += offset
            end_j = start_j + stepy
            if end_j > height:
                end_j = height

            if (end_i - start_i) > 0 and (end_j - start_j) > 0:
                if use_median:
                    luminance = np.median(img_array[start_i:end_i, start_j:end_j])
                else:
                    luminance = np.mean(img_array[start_i:end_i, start_j:end_j])

                luminance_selector = int(np.round(luminance * max_gray_levels))

                if np.random.rand() > dropout:
                    img_phosphenes[start_i:end_i, start_j:end_j] = phosphenes_look_up[0:end_i-start_i, 0:end_j-start_j, luminance_selector]



    return img_phosphenes


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt

    path = 'your path'
    img = Image.open(path).convert('L')
    img_array = np.asarray(img) / 255
    np.random.seed(10)

    img_phosphenes = convert_phosphenes(img_array, n_phosphenes_x=32, n_phosphenes_y=32, use_median=True, use_hex_pattern=False, dropout=0.1)home/melani/Desktop/vv/Office_1.png

    plt.imshow(img_array)
    plt.figure()
    plt.imshow(img_phosphenes)
    plt.show()


