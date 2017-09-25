from __future__ import print_function

import os, gc, re, sys, glob, cv2, h5py, codecs
import numpy as np
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True



import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# Image batch generator
#   img, image path
#   label, image label
class dogloader(Dataset):
    def __init__(self, img, label, transform = None):
        self.img = img; self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        idx_img = Image.open(self.img[idx]).convert('RGB')
        if self.transform is not None:
            idx_img = self.transform(idx_img)
        label = torch.from_numpy(np.array([self.label[idx]]))
        return idx_img, label

# Normal batch generator
#   x and y are list or array
class Arrayloader(Dataset):
    def __init__(self, x, y):
        self.x = x; self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        idx_x = torch.from_numpy(np.array([self.x[idx]]))
        print(self.y.shape)
        idx_y = torch.from_numpy(np.array(self.y[idx],dtype=np.int64))
        return idx_x, idx_y

