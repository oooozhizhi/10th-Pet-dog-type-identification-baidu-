import numpy as np
import os
import shutil
import h5py
import random








for dir in os.listdir(r"/home/deeplearning/wh/baiduImage/train_cut"):
	print(dir)
	for filename in os.listdir("/home/deeplearning/wh/baiduImage/train_cut/"+dir):
	    if os.path.exists("/home/deeplearning/wh/baiduImage/train_cutpre/"+dir)==False:
	         os.mkdir("/home/deeplearning/wh/baiduImage/train_cutpre/"+dir)
	         os.mkdir("/home/deeplearning/wh/baiduImage/val_cutpre/"+dir)
	    if random.randint(0, 5)!=0:	    	
	   		shutil.copy("/home/deeplearning/wh/baiduImage/train_cut/"+dir+"/"+filename,"/home/deeplearning/wh/baiduImage/train_cutpre/"+dir+"/"+filename)
	    else:
	    	shutil.copy("/home/deeplearning/wh/baiduImage/train_cut/"+dir+"/"+filename,"/home/deeplearning/wh/baiduImage/val_cutpre/"+dir+"/"+filename)




