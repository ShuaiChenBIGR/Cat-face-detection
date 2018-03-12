#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:03:10 2018

@author: seasar
"""
import os
import Modules.Common_modules as cm
import cv2
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from sklearn.model_selection import train_test_split
from keras.applications import xception
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras import backend as K
from tqdm import tqdm  
np.random.seed(111)


print(os.listdir("/hdd2/PythonCodes/Git_clone/seaser/data/"))

input_path = '/hdd2/PythonCodes/Git_clone/seaser/data/'
cats = os.listdir(input_path)
print("Total number of sub-directories found: ", len(cats))

# Store the meta-data in a dataframe for convinience 
data = []
for folder in cats:
    new_dir = Path(input_path + folder)
    images = sorted(new_dir.glob('*.jpg'))
    annotations = sorted(new_dir.glob('*.cat'))
    n = len(images)
    for i in range(n):
        img = str(images[i])
        annotation = str(annotations[i])
        data.append((img, annotation))
    print("Processed: ", folder)
print(" ")        
        
df = pd.DataFrame(data=data, columns=['img_path', 'annotation_path'], index=None)
print("Total number of samples in the dataset: ", len(df))
print(" ")
df.head(10)

#imgPath = []
#imgDescri = []
#for root, dirs, files in os.walk(r"/media/seasar/资源/BaiduYunDownload/cats"):
#    for file in files:
#        if os.path.splitext(file)[1] == ".jpg":
#            imgPath.append(os.path.join(root, file))

def read_img(img_path, train_or_test, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    img = image.load_img(img_path)
    (w,h) = img.size
    img = img.resize(size)
    img = image.img_to_array(img)
    return img,w,h

INPUT_SIZE = 299
POOLING = 'avg'
x_train = np.zeros((len(df), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
y_train = np.zeros((len(df), 18), dtype='float32')
for i, img_path in tqdm(enumerate(df['img_path'])):
    img,w,h = read_img(img_path, 'train', (INPUT_SIZE, INPUT_SIZE))
    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    x_train[i] = x
    
    f = open(df['annotation_path'][i])
    points = f.read().split(' ')
    points = [int(z) for z in points if z!='']
    # Get the list of x and y coordinates
    xpoints = points[1:19:2]
    xpoints = np.array(xpoints)
    ypoints = points[2:19:2]
    ypoints = np.array(ypoints)
    xpoints = xpoints*INPUT_SIZE/w
    ypoints = ypoints*INPUT_SIZE/h
    point = np.array([xpoints,ypoints])
    point = point.T
    point = point.reshape([1,-1])
    y_train[i] = point 
print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

np.save(cm.workingPath.home_path +'xtrain.npy', x_train)
np.save(cm.workingPath.home_path +'ytrain.npy', y_train)