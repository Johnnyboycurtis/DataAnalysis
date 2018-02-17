#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:15:48 2018

@author: jonathan
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation


df = pd.read_csv("../data/mcycle.csv")

vals = df['accel']
X = df['times']

def scale(x):
    mu = np.median(x)
    sd = np.std(x)
    return (x - mu)/sd

X = scale(X)



def tilted_loss(q,y,f):
    """
    f(u) =  \tau * u if u > 0, else (\tau - 1) * u
         = u (\tau - I(u < 0))
    Reference: http://www.lokad.com/pinball-loss-function-definition
    """
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e))



model = Sequential()
model.add(Dense(units=10, input_dim=1, activation='tanh'))
model.add(Dense(units=10, input_dim=1, activation='relu'))
model.add(Dense(1))
model.compile(loss=lambda y,f: tilted_loss(0.5,y,f), optimizer='adadelta')
model.fit(x = X, y = vals, epochs = 3000, verbose=0)

out = model.predict(X)
#out = list(zip(out, vals))
#[print(z) for z in out]



plt.scatter(x = X, y = vals)
plt.scatter(x = X, y = out, marker = "x")
plt.show()
time.sleep(1)
plt.close('all')



del model

df = pd.read_csv("../data/co2.csv")

vals = df['x']
X = scale(df.index)
#X = df.index.values ## takes much longer to converge without scaling


model = Sequential()
model.add(Dense(units=10, input_dim=1, activation='relu'))
model.add(Dense(units=10, input_dim=1, activation='tanh'))
model.add(Dense(1))
model.compile(loss=lambda y,f: tilted_loss(0.5,y,f), optimizer='adadelta')
model.fit(x = X, y = vals, epochs = 2000, verbose=0)

out = model.predict(X)
plt.plot(X, vals)
plt.plot( X, out, "--")
plt.ylim((300, 380))
plt.title("Atmospheric concentration of CO")
plt.show()
time.sleep(2)
plt.close('all')



