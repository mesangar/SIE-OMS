import argparse
import os
import numpy as np
import tensorflow as tf
import scipy.misc
from scipy import misc
from matplotlib import pyplot as plt
from PIL import Image
import glob
import time

import models

def predict(model_data_path, image_path_list):

    
    # Default input size
    height = 256
    width = 320
    channels = 3
    batch_size = 1
   
    
    # Create a placeholder for the input image
    rgb_ph = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.EdgeEstimator({'rgb_input':rgb_ph}, batch_size, trainable = False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from npy file
        net.load(model_data_path, sess) 

        # Evalute the network for the given image
        for image_path in image_path_list:
            img = Image.open(image_path)
            img = img.resize([width,height], Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = np.expand_dims(np.asarray(img), axis = 0)
   
            fd = net.fd_test
        fd[rgb_ph] = img
        pred = sess.run(tf.nn.sigmoid(net.get_layer_output("edge_likelihood")), feed_dict=fd)
		        
		# Plot result
		#fig = plt.figure()
		#ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
		#fig.colorbar(ii)
		#plt.show()
		
		# Save result
        name=str(image_path)
		#scipy.misc.imsave("/home/clara/workspace/hedau+_data/EM/"+name[43:-4]+"_emap.jpg", pred[0,:,:,0])
        scipy.misc.imsave("your path"+"sample_emap.jpg", pred[0,:,:,0])  # CAMBIAR: DIRECCION DONDE QUIERES QUE TE GUARDE LOS IMAGENES RESULTANTES 
               
        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument('model_path', help='Converted parameters for the model')
    args = parser.parse_args()

    # Predict the image
    t = time.time()
    #for i_rgb in glob.glob("/home/clara/workspace/hedau+_data/test_new/*.jpg"):          
    #    tf.reset_default_graph()
    #pred = predict(args.model_path, glob.glob("/home/clara/workspace/hedau+_data/test_new/*.jpg")) 
    pred = predict("single-view120.npy", glob.glob("your path/*.jpg")) # CAMBIAR: DIRECCION DE DONDE COGE LAS IMAGENES RGB 
    #args.model_path
    elapsed = time.time() - t
    print('elapsed/1',elapsed/1)		
    
    os._exit(0)

if __name__ == '__main__':
    main()
    
    
#usage: python predict.py single-view120.npy 

    

