import numpy as np
import os
import shutil
import h5py
import random
import inceptionV4

from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img

datagen = ImageDataGenerator(rotation_range=40,
                     		 shear_range=0.2,                    		
                             zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')




print("2p")
for dir in os.listdir(r"/home/deeplearning/wh/baiduImage/train_cut_hun"):
	print(dir)
	for filename in os.listdir("/home/deeplearning/wh/baiduImage/train_cut_hun/"+dir):
	    if os.path.exists("/home/deeplearning/wh/baiduImage/train_cut2p_hun/"+dir)==False:
	         os.mkdir("/home/deeplearning/wh/baiduImage/train_cut2p_hun/"+dir)
	    shutil.copy("/home/deeplearning/wh/baiduImage/train_cut_hun/"+dir+"/"+filename,"/home/deeplearning/wh/baiduImage/train_cut2p_hun/"+dir+"/"+filename)
	    img = load_img("/home/deeplearning/wh/baiduImage/train_cut2p_hun/"+dir+"/"+filename)
	    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	    x = x.reshape((1,) + x.shape)  
	    i=0
	    for batch in datagen.flow(x, batch_size=1,save_to_dir="/home/deeplearning/wh/baiduImage/train_cut2p_hun/"+dir+"/", save_prefix='dog', save_format='jpg'):
	        i += 1
	        if i > 0:
	            break  
