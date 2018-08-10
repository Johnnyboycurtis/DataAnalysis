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


xinput = Input(shape=(32, 32, 3))
conv1 = Conv2D(filters=64, kernel_size=(2,2), strides=1)(xinput)
conv1 = MaxPooling2D(pool_size=(3,3), strides=1)(conv1)

ident = conv1*1
conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=1)(conv1)
conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=1)(conv2)
conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=1)(conv2) # + conv1
conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=1)(conv2)

conv3 = Conv2D(filters=128, kernel_size=(2,2), strides=1)(conv2)
conv3 = Conv2D(filters=128, kernel_size=(2,2), strides=1)(conv3)
conv3 = Conv2D(filters=128, kernel_size=(2,2), strides=1)(conv3) # + conv2
conv3 = Conv2D(filters=128, kernel_size=(2,2), strides=1)(conv3)

conv4 = Conv2D(filters=256, kernel_size=(1,1), strides=1)(conv3)
conv4 = Conv2D(filters=256, kernel_size=(1,1), strides=1)(conv4)
conv4 = Conv2D(filters=256, kernel_size=(1,1), strides=1)(conv4) # + conv1
conv4 = Conv2D(filters=256, kernel_size=(1,1), strides=1)(conv4)

conv5 = Conv2D(filters=512, kernel_size=(1,1), strides=1)(conv4)
#conv5 = Conv2D(filters=512, kernel_size=(2,2), strides=1)(conv5)
#conv5 = Conv2D(filters=512, kernel_size=(2,2), strides=1)(conv5) # + conv1
#conv5 = Conv2D(filters=512, kernel_size=(2,2), strides=1)(conv5)

xout = MaxPooling2D(pool_size=(3,3), strides=2)(conv5)
xout = Flatten()(xout)
xout = Dense(num_classes, activation='softmax')(xout)


model = Model(xinput, xout)
print(model.summary())

from keras.metrics import categorical_accuracy
model.compile(loss='categorical_crossentropy', optimizer=Adadelta, metrics=[categorical_accuracy])


# Train the model
start = time.time()
model_info = model.fit(train_features, train_labels, 
                       batch_size=32, epochs=20, 
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