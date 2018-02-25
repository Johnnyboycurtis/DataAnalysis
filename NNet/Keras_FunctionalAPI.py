import numpy as np
import pandas as pd
import patsy
from keras.models import Model
from keras.layers import Input, Dense


df = pd.read_csv("../data/mtcars.csv")

y, X = patsy.dmatrices(data=df, formula_like='mpg ~ am + wt')

m = X.shape[1] ## columns
#print(X)

l0 = Input(shape = (m, )) ## input layer is the only layer
l1 = Dense(1)(l0) ## output
model = Model(inputs=l0, outputs = l1)
model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(y=y, x=X, epochs = 50)

