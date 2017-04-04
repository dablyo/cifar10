__author__ = 'wang'

import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import shutil
import StringIO
#-----------------------------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import mnist
from PIL import Image
import os
import os.path
import random
import cv2
import math
from scipy import ndimage
import torch.utils.data as data
#-----------------------------------------------------------------------------------
classes=['0','1','2','3','4','5','6','7','8','9']
root='MNIST'
srcdir='raw'
dstdir='mnistpic'
trainimgfile='train-images-idx3-ubyte'
trainlabelfile='train-labels-idx1-ubyte'
testimgfile='t10k-images-idx3-ubyte'
testlabelfile='t10k-labels-idx1-ubyte'

#-----------------------------------------------------------------------------------

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b

def read_label_file(file):
    with open(file,'rb') as f:
        data=f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return labels


def read_image_file(file):
    with open(file, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            print l
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return length,num_rows,num_cols,images

#-----------------------------------------------------------------------------------
#%matplotlib inline
def imshow(img):
    img = img / 2 + 0.5 # unnormalize Inception v3 	22.55 	6.44
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)
    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx,shifty
    
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted    
  
def centrelize(gray): # move a char, black as background, and white as foreground,  to center of a 28*28 image
    print gray.shape
    print gray[0]
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
        
    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
        
    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape
        
    if rows>cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray,(cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray,(cols,rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    return shifted    

class PICSET(data.Dataset):
    picroot='pic'

    def __init__(self,root,transform=None):
        self.picroot=root
        self.transform=transform
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        
        if not os.path.exists(self.picroot):
            raise RuntimeError('{} doesnot exists'.format(self.picroot))
        for root,dnames,filenames in os.walk(self.picroot):
            imgs=np.ndarray(shape=(len(filenames),28,28),dtype=np.uint8)
            i=0
            for filename in filenames:
                picfilename=os.path.join(self.picroot,filename)
                im=cv2.imread(picfilename,cv2.IMREAD_GRAYSCALE)
                im=cv2.resize(255-im,(28,28))
                (thresh, im) = cv2.threshold(im, 32, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)   
                #im=cv2.erode(im,self.kernel)
                #im=cv2.erode(im,self.kernel)
                #im=cv2.dilate(im,self.kernel)
                #print im.shape
                #im=cv2.GaussianBlur(im,(5,5),0.1)
                imgs[i]=centrelize(gray=im)
                #imgs[i]=cv2.resize(im,(28,28))
                i=i+1
            self.dataset=torch.ByteTensor(imgs)
            self.len=len(filenames)       
    
    def __getitem__(self,index):
        img=self.dataset[index]
        img=Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img=self.transform(img)
        return img
    
    def __len__(self):
        return self.len


#-----------------------------------------------------------------------------------

if __name__ == '__main__':
    batch=1
    transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                             ])
    picset = PICSET(root='pic', transform=transform)
    picloader = torch.utils.data.DataLoader(picset, batch_size=batch, shuffle=False, num_workers=1)
    piciter = iter(picloader)

    exit()

    #cs, cti = find_classes('/home/wang/git/cifar10/MNIST')
    if os.path.exists(os.path.join(root,dstdir)):
        shutil.rmtree(os.path.join(root,dstdir))
    os.mkdir(os.path.join(root,dstdir))
    for i in range(10):
        os.mkdir(os.path.join(root,dstdir,classes[i]))
    filename=os.path.join(root, srcdir,testlabelfile)
    labels=read_label_file(filename)
    filename=os.path.join(root, srcdir, testimgfile)
    num_pic,num_h,num_v,imgs=read_image_file(filename)
    assert len(labels)==num_pic
    assert num_h==len(imgs[0])
    assert num_v==len(imgs[0][0])
    for i in range(num_pic):
        im=Image.open(StringIO.StringI(imgs[i]))
        im.save(os.path.join(root,dstdir,'{}/{}'.format(labels[i],i),'jpg'))
        print os.path.join(root,dstdir,'{}/{}'.format(labels[i],i),'jpg')
