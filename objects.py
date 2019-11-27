import numpy as np
from scipy.ndimage import binary_erosion
from scipy.misc import imresize
def enhance_masks(
        image,
        all_class_names,
        used_class_names,   # ['table', 'chair']
        results,
        background=None
):
    print(type(image))
    image = imresize(image,(480,480,3))

    print(image.shape)

    if background is None:
        new_image = np.zeros((image.shape[0],image.shape[1]))

    else:
        new_image = background


    object_list_ids = reversed(range(results['class_ids'].shape[0]))
    print(results['class_ids'].shape)

    result = []
    for i in range(results['class_ids'].shape[0]):
        _im = results['masks']
        result.append(imresize(_im[:,:,i], (image.shape[0],image.shape[1],1)))

    result = np.array(result)
    print(result.shape)


    for object_id in object_list_ids:

        if all_class_names[results['class_ids'][object_id]] in used_class_names:

            filling = binary_erosion(result[object_id,...], structure=None,
                            iterations=14) * 0.5
            print(filling.shape)

            new_object = result[object_id,...] - filling
            print(new_object.shape)
            colored_pixels = new_object > 0
            new_image[colored_pixels] = new_object[colored_pixels]
    return new_image
