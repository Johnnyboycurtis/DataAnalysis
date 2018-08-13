#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:43:17 2018

@author: jn107154
"""

import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation, Add
from keras.models import Model


import time
import numpy as np
from keras.utils import np_utils

np.random.seed(2017)

#import cifar10
from keras.datasets import cifar10
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  test_features.shape
num_classes = len(np.unique(train_labels))

print("dimensions", train_features.shape)

ind = np.random.rand(50000) < 0.85
val_features = train_features[~ind]
val_labels = train_labels[~ind]

train_features = train_features[ind]
train_labels = train_labels[ind]

ind = np.random.rand(10000) < 0.30
test_labels = test_labels[ind]
test_features = test_features[ind]


train_features = train_features.astype('float32')/255
val_features = val_features.astype('float32')/255
test_features = test_features.astype('float32')/255
# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
val_labels = np_utils.to_categorical(val_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)



Adadelta = keras.optimizers.Adadelta()

print('building regression model')
xinput = Input(shape=(32, 32, 3))
l1 = Conv2D(filters=64, kernel_size=(3,3), strides=2)(xinput)
mp1 = MaxPooling2D(pool_size=(2,2), strides=2)(l1)
l2 = Conv2D(filters=64, kernel_size=(2,2), strides=2, padding='same')(mp1)
l3 = Conv2D(filters=64, kernel_size=(2,2), strides=2, padding='same')(l2)
z = Add()([mp1, l3])
l4 = Conv2D(filters=64, kernel_size=(2,2), strides=2, padding='same')(z)
l5 = Conv2D(filters=64, kernel_size=(2,2), strides=2, padding='same')(l4)
z = Add()([l3, l5])
l6 = Conv2D(filters=128, kernel_size=(2,2), strides=2, padding='same')(z)
l7 = Conv2D(filters=128, kernel_size=(2,2), strides=2, padding='same')(l6)
B = Conv2D(filters=128, kernel_size=(1,1), strides=2)(l5) # project to match sizes
z = Add()([B, l7]) # can pass
l8 = Conv2D(filters=128, kernel_size=(2,2), strides=2, padding='same')(l7)
l9 = Conv2D(filters=128, kernel_size=(2,2), strides=2, padding='same')(l8)
z = Add()([l7, l9])
l10 = Conv2D(filters=256, kernel_size=(1,1), strides=2, padding='same')(z)
l11 = Conv2D(filters=256, kernel_size=(1,1), strides=2, padding='same')(l10)
B = Conv2D(filters=256, kernel_size=(1,1), strides=2)(l9) # projection to match sizes
z = Add()([B, l11]) # can pass
l12 = Conv2D(filters=256, kernel_size=(1,1), strides=2, padding='same')(z)
l13 = Conv2D(filters=256, kernel_size=(1,1), strides=2, padding='same')(l12)
z = Add()([l11, l13])
l14 = Conv2D(filters=512, kernel_size=(1,1), strides=2, padding='same')(z)
l15 = Conv2D(filters=512, kernel_size=(1,1), strides=2, padding='same')(l14)
B = Conv2D(filters=512, kernel_size=(1,1), strides=2)(l13) # projection to match sizes
z = Add()([B, l15]) # can pass
l16 = Conv2D(filters=512, kernel_size=(1,1), strides=2, padding='same')(l15)
l17 = Conv2D(filters=512, kernel_size=(1,1), strides=2, padding='same')(l16)
z = Add()([B, l17])
mp2 = MaxPooling2D(pool_size=(2,2), strides=2)(z)
flat = Flatten()(mp2)
l18 = Dense(10, activation='softmax')(flat)
model = Model(xinput, l18)

print(model.summary())

from keras.metrics import categorical_accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])



# Train the model
start = time.time()
model_info = model.fit(train_features, train_labels, 
                       batch_size=32, epochs=20, 
                       validation_data = (val_features, val_labels), 
                       verbose=1)
end = time.time()
# plot model history
print("Model took %0.2f seconds to train"%(end - start))


