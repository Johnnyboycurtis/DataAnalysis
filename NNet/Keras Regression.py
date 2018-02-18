#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 09:41:11 2018

@author: jonathan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation


def scale(x):
    """
    Scale values; uses median instead of mean
    """
    mu = np.median(x)
    sd = np.std(x)
    return (x - mu)/sd


df = pd.read_csv("../data/mtcars.csv")
y = df.mpg
X = scale(df.wt)

"""
plt.scatter(X, y)
plt.title("MPG ~ WT")
plt.show()
"""


model = Sequential(name = 'Regression')
model.add(Dense(units = 10, input_dim = 1, activation='relu'))
model.add(Dense(units = 10, activation='relu'))
model.add(Dense(1)) ## output layer!
model.compile(loss='mean_squared_error', optimizer='sgd') 


model.fit(x = X, y = y, epochs = 200)

out = model.predict(x = X)

plt.scatter(X, y)
plt.scatter(X, out, marker = "+")
plt.title("MPG ~ WT (optimizer = sgd)")
plt.show()


del model



model = Sequential(name = 'Regression')
model.add(Dense(units = 10, input_dim = 1, activation='relu'))
model.add(Dense(units = 10, activation='relu'))
model.add(Dense(1)) ## output layer!
model.compile(loss='mean_squared_error', optimizer='adadelta') 


model.fit(x = X, y = y, epochs = 200)

out = model.predict(x = X)

plt.scatter(X, y)
plt.scatter(X, out, marker = "+")
plt.title("MPG ~ WT (optimizer = adadelta)")
plt.show()
