import os
os.chdir('/home/jonathan/Documents/DataAnalysis/Image')
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential(name = 'Regression')
model.add(Dense(units = 10, input_dim = 9, activation='relu'))
model.add(Dense(units = 10, activation='relu'))
model.add(Dense(1)) ## output layer!
model.compile(loss='mean_squared_error', optimizer='sgd') 



class DataReader():
    def __init__(self, filename, sep = ',', label_col=0, header = True):
        self.filename = filename
        self.sep = sep
        self.label_col = label_col
        self.header = header
    
    def __iter__(self):
        filename = self.filename
        sep = self.sep
        label_col = self.label_col
        header = self.header
        with open(filename) as myfile:
            if header:
                _ = myfile.readline()
            for line in myfile:
                y, x = parse(line, sep, label_col)
                yield x, y    
    
                
                
def parse(line, sep, label_col):
    line = line.strip().split(sep)
    y = np.float32(line[label_col])
    x = np.float32(line[label_col:])
    return x, y
    


def generate_arrays_from_file(path='pima.csv'):
    while 1:
        f = open(path)
        f.readline()
        for line in f:
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x, y = parse(line, sep=',', label_col=0)
            y = np.array(y).reshape((1,))
            x = x.reshape((1,9))
            print(y, x)
            yield (x, y)
        f.close()



import math
import keras
import matplotlib.pyplot as plt

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

        return np.array([ plt.imread(file_name)  for file_name in batch_x ]), np.array(batch_y)





if __name__ == '__main__':
    #data = generate_arrays_from_file()
    #model.fit_generator(data, steps_per_epoch=20, epochs=4, verbose=0)
    with open('ImagePaths.csv') as myfile:
        x_set = [path.strip() for path in myfile]
    y_set = pd.read_csv('../data/Digits/train.csv').iloc[:,0]
    img_seq = ImgSequence(x_set, y_set, batch_size=10)
    model.fit_generator(img_seq, epochs=4, verbose=0)
    
    
