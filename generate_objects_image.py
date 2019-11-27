import objects
from scipy import ndimage, misc
import os
import sys
import numpy as np
import coco
import utils
import model as modellib
from PIL import Image

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "/home/melani/Desktop/Mask_RCNN/mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "/home/melani/Desktop/Ba")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO dataset contains 82 objects categories with more than 5000 instance each.
# COCO Class names
# Index of the class in the list is its ID.

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

#used_class_names = ['tv', 'chair', 'dining table', 'laptop']
used_class_names = ['toilet', 'sink']
#used_class_names = ['bed']
#used_class_names = ['chair', 'dining table']
#used_class_names = ['oven', 'tv', 'microwave', 'refrigerator', 'sink']
#used_class_names = ['chair', 'dining table', 'couch']


for image_path in sorted(os.listdir(IMAGE_DIR)):
    print(image_path)
    input_path = os.path.join(IMAGE_DIR, image_path)
    print(input_path)
    image = ndimage.imread(input_path)

    print(image.shape)
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    print(r)




    New_objects_max = objects.enhance_masks(image, class_names, used_class_names, r)


    New_objects_maxi = New_objects_max * 255
    print(np.amax(New_objects_maxi))
    fullpath = os.path.join("/home/melani/Desktop/Mask/", 'Mask_' + image_path)
    misc.imsave(fullpath, New_objects_max)