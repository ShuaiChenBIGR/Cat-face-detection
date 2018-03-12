#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:03:10 2018

@author: seasar
"""
import os
import Modules.Common_modules as cm
from keras import callbacks
import cv2
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input, GlobalAveragePooling2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
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

cm.mkdir(cm.workingPath.model_path)
cm.mkdir(cm.workingPath.best_model_path)

x_train = np.load(cm.workingPath.home_path +'xtrain.npy')
y_train = np.load(cm.workingPath.home_path +'ytrain.npy')




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

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
#
# # compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='mse')
#
# # train the model on the new data for a few epochs
model.fit(x_train,y_train, batch_size=32, epochs=2, verbose=1,validation_split=0.1)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:(不冻了 all True)
for layer in model.layers[:249]:
   layer.trainable = True
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use Adam with a low learning rate
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='mse')

bestfilepath = cm.workingPath.model_path + 'Best_weights.{epoch:02d}-{loss:.5f}.hdf5'
model_best_checkpoint = callbacks.ModelCheckpoint(bestfilepath, monitor='loss', verbose=0, save_best_only=True)

callbacks_list = [model_best_checkpoint]
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(x_train,y_train, batch_size=32, epochs=500, verbose=1, validation_split=0.1, callbacks=callbacks_list)

print('training finished')
