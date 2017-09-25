import numpy as np
import os
import shutil
import h5py
import random








for dir in os.listdir(r"/your path/train_cut"):
	print(dir)
	for filename in os.listdir("/your path/train_cut/"+dir):
	    if os.path.exists("/your path/train_cutpre/"+dir)==False:
	         os.mkdir("/your path/train_cutpre/"+dir)
	         os.mkdir("/your path/val_cutpre/"+dir)
	    if random.randint(0, 5)!=0:	    	
	   		shutil.copy("/your path/train_cut/"+dir+"/"+filename,"/your path/train_cutpre/"+dir+"/"+filename)
	    else:
	    	shutil.copy("/your path/train_cut/"+dir+"/"+filename,"/your path/val_cutpre/"+dir+"/"+filename)




