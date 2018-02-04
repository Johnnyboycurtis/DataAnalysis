#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, product
import pandas as pd
import patsy
from scipy.spatial import distance
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
#from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Import some data to play with
iris = pd.read_csv("/home/jonathan/Documents/iris.csv")
iris.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

y, X = patsy.dmatrices(data=iris, formula_like="Species ~ SepalLength + SepalWidth + PetalLength + PetalWidth")

# some weights
w = {1: 1.0, 2: 1.9, 3: 1.1}

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)


# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = svm.LinearSVC(C= 7, dual=False, class_weight=w)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

y_mat_test = label_binarize(y_test, classes=[1, 2, 3])

# Compute ROC curve and ROC area for each class
n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()
optim_thres = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i]  = roc_curve(y_mat_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    TOP = (0,1)
    f = fpr[i]
    t = tpr[i]
    thres = thresholds[i]
    dist = []
    for j in range(len(thres)):
        coord = (f[j], t[j])
        d = distance.euclidean(TOP, coord)
        dist.append(d)
    ind = np.argmin(dist)
    optim_thres[i] = (f[ind], t[ind], thres[ind])




for i in range(n_classes):
    plt.figure()
    lw = 2
    plt.plot(fpr[i], tpr[i], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.scatter(fpr[i], tpr[i])
    a,b, label = optim_thres[i]
    label = "threshold: {}".format(round(label,4))
    plt.annotate(
        label,
        xy=(a, b), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    #break











# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(y_mat_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])





# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()














from sklearn.metrics import confusion_matrix

y_pred = classifier.fit(X_train, y_train).predict(X_test)

#class_names = {1: "setosa",  2:"versicolor", 3:"virginica"}
class_names = ["setosa",  "versicolor", "virginica"]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()









# some results from choosing to use optimal threshold
# you lose some accuracy, but gain a drop in false positives
results = []
for i in range(n_classes):
    yhat = y_score[:, i]
    ytest = y_mat_test[:, i]
    a, b, test = optim_thres[i]
    vals = (yhat >= test).astype(float)
    results.append(vals)
    print(np.mean(vals == ytest))


## please refer to: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1444894/



