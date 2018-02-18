#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:33:28 2018

@author: jonathan
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

def scale(x):
    """
    Scale values; uses median instead of mean
    """
    mu = np.median(x)
    sd = np.std(x)
    return (x - mu)/sd


df = pd.read_csv("../data/co2.csv")


vals = df['x']
X = scale(df.index)



#plt.plot(df['x'])
#plt.title("Atmospheric concentration of CO2")
#plt.show()


model = Sequential(name = 'TimeSeries')
model.add(Dense(units = 10, input_dim = 1, activation='tanh'))
model.add(Dense(units = 5, input_dim = 1, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adadelta', loss = 'mean_squared_error', ) ## faster convergence
#model.compile(optimizer='rmsprop', loss = 'mean_squared_error')
#model.compile(optimizer = 'sgd', loss = 'mean_absolute_error')

model.fit(x = X, y = vals, epochs=600, verbose=0)


out = model.predict(X)
plt.plot(X, vals)
plt.plot( X, out, "--")
plt.ylim((300, 380))
plt.title("Atmospheric concentration of CO")
plt.show()


