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
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input, GlobalAveragePooling2D,Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from sklearn.model_selection import train_test_split
from keras.applications import xception
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras import backend as K
from tqdm import tqdm

print(os.listdir("/hdd2/PythonCodes/Git_clone/seaser/data/"))

input_path = '/hdd2/PythonCodes/Git_clone/seaser/data/'
cats = os.listdir(input_path)
print("Total number of sub-directories found: ", len(cats))

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
  (w, h) = img.size
  img = img.resize(size)
  img = image.img_to_array(img)
  return img, w, h

INPUT_SIZE = 299

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
# and a logistic layer -- let's say we have 18 points
predictions = Dense(18, activation='relu')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights(cm.workingPath.model_path +'Best_weights.479-29.74871.hdf5')

# test_path = "/hdd2/PythonCodes/Git_clone/seaser/data/CAT_00/00000003_013.jpg"
# test_path = "/hdd2/PythonCodes/Git_clone/seaser/data/ShuaiChen.jpg"
test_path = "/hdd2/PythonCodes/Git_clone/seaser/data/Selection_078.png"
cat_test,w_test,h_test = read_img(test_path, 'test', (INPUT_SIZE, INPUT_SIZE))
cat_test_x = xception.preprocess_input(np.expand_dims(cat_test.copy(), axis=0))
y_test = model.predict(cat_test_x)
xpoints_test = y_test[:,0:18:2]
ypoints_test = y_test[:,1:18:2]
plt.imshow(image.load_img(test_path,target_size=(INPUT_SIZE,INPUT_SIZE)))
plt.scatter(xpoints_test, ypoints_test, c='g')
print(xpoints_test)
print(ypoints_test)
plt.show()