#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 14:58:20 2018

@author: jn107154
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:43:17 2018

@author: jn107154
"""

import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation #, merge, AveragePooling2D
from keras.models import Model
from keras.metrics import categorical_accuracy



import time
import matplotlib.pyplot as plt
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
#rmsprop = keras.optimizers.RMSprop(lr=0.001995, rho=0.95, epsilon=None, decay=0.001)



print('building regression model')


class ResNet18:
    def __init__(self):
        num_classes = 10
        self.conv1 = Conv2D(filters=64, kernel_size=(1,1), strides=1)
        self.MP1 = MaxPooling2D(pool_size=(3,3), strides=1) # skip from here
        self.conv2 = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same')
        self.conv3 = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same')
        self.conv4 = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='same')
        self.conv5 = Conv2D(filters=512, kernel_size=(1,1), strides=1)
        self.flatten = Flatten()
        self.MP2 = MaxPooling2D(pool_size=(3,3), strides=2) # replaced avg with max pool
        self.softmax = Dense(num_classes, activation='softmax')
        
        self.build()
        
    def build(self):
        xinput = Input(shape=(32, 32, 3))
        l1 = self.conv1(xinput)
        mp1 = self.MP1(l1)
        l2 = self.conv2(mp1)
        l3 = self.conv2(l2)
        l4 = self.conv2(l3)
        l5 = self.conv2(l4)
        l6 = self.conv3(l5)
        l7 = self.conv3(l6)
        l8 = self.conv3(l7)
        l9 = self.conv3(l8)
        l10 = self.conv4(l9)
        l11 = self.conv4(l10)
        l12 = self.conv4(l11)
        l13 = self.conv4(l12)
        l14 = self.conv5(l13)
        l15 = self.conv5(l14)
        l16 = self.conv5(l15)
        l17 = self.conv5(l16)
        mp2 = self.MP2(l17)
        flat = self.flatten(mp2)
        l18 = self.softmax(flat)
        model = Model(xinput, l18)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
        self.model = model
        
    def summary(self):
        return self.model.summary()
        
    def train(self, x, y):
        model_info = model.fit(train_features, train_labels, 
                       batch_size=32, epochs=20, 
                       validation_data = (val_features, val_labels), 
                       verbose=1)
        return model_info


model = ResNet18()
print(model.summary())



# Train the model
start = time.time()
model_info = model.fit(train_features, train_labels, 
                       batch_size=32, epochs=200, 
                       validation_data = (val_features, val_labels), 
                       verbose=1)
end = time.time()
# plot model history
print("Model took %0.2f seconds to train"%(end - start))



'''
xinput = Input(shape=(32, 32, 3))
conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=2)(xinput)
conv1 = MaxPooling2D(pool_size=(3,3), strides=2)(conv1)

conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=2)(conv1)
conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=2)(conv2)
conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=2)(conv2) # + conv1
conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=2)(conv2)

conv3 = Conv2D(filters=128, kernel_size=(3,3), strides=2)(conv2)
conv3 = Conv2D(filters=128, kernel_size=(3,3), strides=2)(conv3)
conv3 = Conv2D(filters=128, kernel_size=(3,3), strides=2)(conv3) # + conv2
conv3 = Conv2D(filters=128, kernel_size=(3,3), strides=2)(conv3)

conv4 = Conv2D(filters=256, kernel_size=(3,3), strides=2)(conv3)
conv4 = Conv2D(filters=256, kernel_size=(3,3), strides=2)(conv4)
conv4 = Conv2D(filters=256, kernel_size=(3,3), strides=2)(conv4) # + conv1
conv4 = Conv2D(filters=256, kernel_size=(3,3), strides=2)(conv4)

conv5 = Conv2D(filters=512, kernel_size=(3,3), strides=2)(conv4)
conv5 = Conv2D(filters=512, kernel_size=(3,3), strides=2)(conv5)
conv5 = Conv2D(filters=512, kernel_size=(3,3), strides=2)(conv5) # + conv1
conv5 = Conv2D(filters=512, kernel_size=(3,3), strides=2)(conv5)
'''