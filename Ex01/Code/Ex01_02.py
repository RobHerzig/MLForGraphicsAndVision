import os
import gzip
import numpy as np

def load_mnist(path, kind='train', each=1):
    labels_path = os.path.join(path, '%s−labels−idx1−ubyte.gz' % kind)
    images_path = os.path.join(path, '%s−images−idx3−ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    images = images[::each, :]
    labels = labels[::each]

    return images, labels

mnist_path_train = r'C:/Users/Rob/Documents/MachineLearningForGraphicsAndVision/downloads/'
kind_train = 'train'
kind_test = 't10k'

load_mnist(mnist_path_train, kind=kind_train)
