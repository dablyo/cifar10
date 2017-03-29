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

classes=['0','1','2','3','4','5','6','7','8','9']
root='MNIST'
srcdir='raw'
dstdir='mnistpic'
trainimgfile='train-images-idx3-ubyte'
trainlabelfile='train-labels-idx1-ubyte'
testimgfile='t10k-images-idx3-ubyte'
testlabelfile='t10k-labels-idx1-ubyte'

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



if __name__ == '__main__':
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
