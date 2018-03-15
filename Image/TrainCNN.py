import os
os.chdir('/home/jonathan/Documents/DataAnalysis/Image')
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
import math
import keras
from sklearn import preprocessing
import matplotlib.pyplot as plt


model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28, 4)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
## fully connected dense layer
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                                    metrics=['accuracy'])

#model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=1)



# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class ImgSequence(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.array([ plt.imread(file_name)  for file_name in batch_x ])
        return X, np.array(batch_y)





if __name__ == '__main__':
    #data = generate_arrays_from_file()
    #model.fit_generator(data, steps_per_epoch=20, epochs=4, verbose=0)
    with open('ImagePaths.csv') as myfile:
        x_set = [path.strip() for path in myfile]
    y_set = pd.read_csv('../data/Digits/train.csv').iloc[:,0]
    y_set = preprocessing.label_binarize(y_set, y_set.unique())
    img_seq = ImgSequence(x_set, y_set, batch_size=20)
    model.fit_generator(img_seq, epochs=4, verbose=1, , shuffle=True)
    
    
