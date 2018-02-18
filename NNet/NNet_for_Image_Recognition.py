#!/usr/bin/python3

"""
This example is taken from Python Machine Learning by Raschka

"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path, kind = 'train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte', % (kind))
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                                imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)
    return images, labels


def example_plot():
    fix, ax = plt.subplots(2,5,sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28,28)
        ax[i].imshow(img, cmap = 'Greys', interpolation='nearest')
    ## set ticks later ##ax[0].set_xticks([])
    plt.tight_layout()
    plt.show()



