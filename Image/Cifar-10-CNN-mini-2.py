# If using tensorflow, set image dimensions order
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")


import time
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

np.random.seed(2017)

#import cifar10
from keras.datasets import cifar10
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  test_features.shape
num_classes = len(np.unique(train_labels))

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(train_labels[:]==i)[0]
    features_idx = train_features[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num,::], (1, 2, 0))
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()


#ind = np.random.rand(50000) < 0.35
#val_features = train_features[~ind]
#val_labels = train_labels[~ind]

#train_features = train_features[ind]
#train_labels = train_labels[ind]

#ind = np.random.rand(10000) < 0.30
#test_labels = test_labels[ind]
#test_features = test_features[ind]


train_features = train_features.astype('float32')/255
#val_features = val_features.astype('float32')/255
test_features = test_features.astype('float32')/255
# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
#val_labels = np_utils.to_categorical(val_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)



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



# Define the model
model = Sequential()
model.add(Convolution2D(64, (4, 4), padding='same', input_shape=(3, 32, 32)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(64, (2, 2), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model

from keras.metrics import categorical_accuracy
Adadelta = keras.optimizers.Adadelta()

model.compile(loss='categorical_crossentropy', optimizer=Adadelta, metrics=[categorical_accuracy])

print(model.summary())
# Train the model
start = time.time()
model_info = model.fit(train_features, train_labels, 
                       batch_size=32, epochs=100, 
                       validation_data = (test_features, test_labels), 
                       verbose=1)
end = time.time()
# plot model history
plot_model_history(model_info)
print("Model took %0.2f seconds to train"%(end - start))
# compute test accuracy
print("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model))



print('saving model')
# serialize model to JSON
model_json = model.to_json()
with open("model-1.json", "w+") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("model-1.h5")
print("Saved model to disk")

