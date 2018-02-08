#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Import some data to play with
df = pd.read_csv("./data/mtcars.csv", index_col=0)

y, X = patsy.dmatrices(data=df, formula_like='cyl ~ mpg + wt + qsec')
#y.shape = (32,)

model = LDA().fit(y = y, X = X)
tX = model.transform(X= X)




plt.scatter(x = tX[:, 0], y = tX[:, 1], s = 100, marker = 'x', cmap = y, edgecolors = 'green')
plt.show()



rdf = pd.DataFrame({'cyl':y, 'lda0':tX[:, 0], 'lda1':tX[:,1]})
rdf.plot.scatter(x = 'lda0', y = 'lda1', c = 'cyl', marker = 'x', s = 100)


grouped = rdf.groupby(by = 'cyl')
for name, frame in grouped:
    plt.scatter(x = frame.lda0, y = frame.lda1)
plt.show()
