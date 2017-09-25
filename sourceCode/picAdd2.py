import numpy as np
import os
import shutil
import h5py
import random
import inceptionV4

from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img

datagen = ImageDataGenerator(rotation_range=40,
							 width_shift_range=0.2,
      					     height_shift_range=0.2,
                     		 shear_range=0.2,                    		
                             zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')




print("2p")
for dir in os.listdir(r"/your path/train_hun"):
	print(dir)
	for filename in os.listdir("/your path/train_hun/"+dir):
	    if os.path.exists("/your path/train2p_hun/"+dir)==False:
	         os.mkdir("/your path/train2p_hun/"+dir)
	    shutil.copy("/your path/train_hun/"+dir+"/"+filename,"/your path/train2p_hun/"+dir+"/"+filename)
	    img = load_img("/your path/train2p_hun/"+dir+"/"+filename)
	    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	    x = x.reshape((1,) + x.shape)  
	    i=0
	    for batch in datagen.flow(x, batch_size=1,save_to_dir="/your path/train2p_hun/"+dir+"/", save_prefix='dog', save_format='jpg'):
	        i += 1
	        if i > 0:
	            break  
