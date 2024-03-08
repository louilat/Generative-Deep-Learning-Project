import numpy as np
import scipy


def get_data_alpha_digits():
    mat = scipy.io.loadmat('data/binaryalphadigs.mat')
    data_digits = mat['dat'][:10,]
    data_alpha = mat['dat'][10:,]
    return data_digits, data_alpha


def lire_alpha_digits(data, list_index):
    m, n = data.shape
    p, q = data[0, 0].shape
    matrix = np.zeros((len(list_index)*n, p*q))
    for i, index in enumerate(list_index):
        for j in range(n):
            matrix[i*n + j] = data[index, j].reshape(p*q)
    return matrix


def get_mnist():
    data_mnist = np.load('data/mnist.npz')
    X_train = data_mnist['x_train']
    X_test = data_mnist['x_test']
    y_train = data_mnist['y_train']
    y_test = data_mnist['y_test']
    return (
        np.round(X_train.reshape(60000, 28*28)/255),
        np.round(X_test.reshape(10000, 28*28)/255),
        y_train, y_test)
