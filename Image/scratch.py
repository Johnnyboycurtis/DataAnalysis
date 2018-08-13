#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 11:26:48 2018

@author: jn107154
"""


import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation, Add #, merge, AveragePooling2D
from keras.models import Model
from keras.metrics import categorical_accuracy

    
class ResNet18:
    def __init__(self):
        self.build()
        
    def build(self):
        xinput = Input(shape=(224, 224, 3))
        l1 = Conv2D(filters=64, kernel_size=(7,7), strides=2)(xinput)
        mp1 = MaxPooling2D(pool_size=(3,3), strides=2)(l1)
        l2 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(mp1)
        l3 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(l2)
        z = Add()([mp1, l3])
        l4 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(z)
        l5 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(l4)
        z = Add()([l3, l5])
        l6 = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same')(z)
        l7 = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same')(l6)
        B = Conv2D(filters=128, kernel_size=(1,1), strides=1)(l5) # project to match sizes
        z = Add()([B, l7]) # can pass
        l8 = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same')(l7)
        l9 = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same')(l8)
        z = Add()([l7, l9])
        l10 = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(z)
        l11 = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(l10)
        B = Conv2D(filters=256, kernel_size=(1,1), strides=1)(l9) # projection to match sizes
        z = Add()([B, l11]) # can pass
        l12 = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(z)
        l13 = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(l12)
        z = Add()([l11, l13])
        l14 = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')(z)
        l15 = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')(l14)
        B = Conv2D(filters=512, kernel_size=(1,1), strides=1)(l13) # projection to match sizes
        z = Add()([B, l15]) # can pass
        l16 = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')(l15)
        l17 = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')(l16)
        z = Add()([B, l17])
        mp2 = MaxPooling2D(pool_size=(3,3), strides=2)(z)
        flat = Flatten()(mp2)
        l18 = Dense(10, activation='softmax')(flat)
        model = Model(xinput, l18)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
        self.model = model
        
    def summary(self):
        return self.model.summary()
        
    def train(self, x, y):
        model_info = model.fit(x, y, 
                       batch_size=32, epochs=20, 
                       verbose=1)
        return model_info
    
    
model = ResNet18()
print(model.summary())




