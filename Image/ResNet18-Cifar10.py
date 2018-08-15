#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:04:41 2018

@author: jn107154
"""

import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation, Add, BatchNormalization
from keras.models import Model


import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt

np.random.seed(2017)


num_classes=10

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)


## model
Adadelta = keras.optimizers.Adadelta()

print('building regression model')
xinput = Input(shape=(32, 32, 3))
#n1 = BatchNormalization()(xinput)
l1 = Conv2D(filters=64, kernel_size=(2,2), strides=2)(xinput)
mp1 = MaxPooling2D(pool_size=(2,2), strides=2)(l1)
l2 = Conv2D(filters=64, kernel_size=(2,2), strides=1, padding='same')(mp1)
l2 = Dropout(0.10)(l2)
l3 = Conv2D(filters=64, kernel_size=(2,2), strides=1, padding='same')(l2)
z = Add()([mp1, l3])
l4 = Conv2D(filters=64, kernel_size=(2,2), strides=1, padding='same')(z)
l4 = Dropout(0.10)(l4)
l5 = Conv2D(filters=64, kernel_size=(2,2), strides=1, padding='same')(l4)
z = Add()([l3, l5])
l6 = Conv2D(filters=128, kernel_size=(2,2), strides=1, padding='same')(z)
l7 = Conv2D(filters=128, kernel_size=(2,2), strides=1, padding='same')(l6)
B = Conv2D(filters=128, kernel_size=(1,1), strides=1)(l5) # project to match sizes
z = Add()([B, l7]) # can pass
l8 = Conv2D(filters=128, kernel_size=(2,2), strides=1, padding='same')(l7)
l8 = Dropout(0.15)(l8)
l9 = Conv2D(filters=128, kernel_size=(2,2), strides=1, padding='same')(l8)
z = Add()([l7, l9])
l10 = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='same')(z)
l11 = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='same')(l10)
B = Conv2D(filters=256, kernel_size=(1,1), strides=1)(l9) # projection to match sizes
z = Add()([B, l11]) # can pass
l12 = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='same')(z)
l12 = Dropout(0.15)(l12)
l13 = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='same')(l12)
z = Add()([l11, l13])
l14 = Conv2D(filters=32, kernel_size=(1,1), strides=1, padding='same')(z)
l15 = Conv2D(filters=32, kernel_size=(1,1), strides=1, padding='same')(l14)
B = Conv2D(filters=32, kernel_size=(1,1), strides=1)(l13) # projection to match sizes
z = Add()([B, l15]) # can pass
l16 = Conv2D(filters=32, kernel_size=(1,1), strides=1, padding='same')(l15)
l16 = Dropout(0.15)(l16)
l17 = Conv2D(filters=32, kernel_size=(1,1), strides=1, padding='same')(l16)
z = Add()([B, l17])
mp2 = MaxPooling2D(pool_size=(2,2), strides=2)(z)
flat = Flatten()(mp2)
#flat = BatchNormalization()(flat)
l18 = Dense(10, activation='softmax')(flat)
model = Model(xinput, l18)


print(model.summary())

from keras.metrics import categorical_accuracy
model.compile(loss='categorical_crossentropy', optimizer=Adadelta, metrics=[categorical_accuracy])




epochs = 50
# fits the model on batches with real-time data augmentation:
model_info = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs, 
                    validation_data=(x_test, y_test))


def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['categorical_accuracy'])+1),model_history.history['categorical_accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_categorical_accuracy'])+1),model_history.history['val_categorical_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['categorical_accuracy'])+1),len(model_history.history['categorical_accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)



# plot model history
plot_model_history(model_info)

# compute test accuracy
print("Accuracy on test data is: %0.2f"%accuracy(x_test, y_test, model))



print('saving model')
# serialize model to JSON
model_json = model.to_json()
with open("model-1.json", "w+") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("model-1.h5")
print("Saved model to disk")