'''
This module opens the CIFAR10 data, transforms it into
numpy arrays, and then builds and trains a CNN.

'''
import cPickle
import numpy as np


def unpack_data(filename):
    '''
    Unpacks the file named filename as a python dictionary.
    '''
    reader = open(filename, 'rb')
    data_dict = cPickle.load(reader)
    reader.close()

    return data_dict


def load_training():
    '''
    Loads and reshapes the data for the training set
    '''
    X_train = np.empty((0, 32 * 32 * 3))
    y_train = np.empty(1)

    for i in xrange(1, 5):
        fname = '../data/data_batch_{}'.format(i)
        data_dict = unpack_data(fname)

        if i == 1:
            X_train = data_dict['data']
            y_train = data_dict['labels']
        else:
            X_train = np.vstack((X_train, data_dict['data']))
            y_train = np.hstack((y_train, data_dict['labels']))

    X_train = X_train.reshape(
        (X_train.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)

    return (X_train, y_train)


def load_validation():
    '''
    Loads and reshapes the data for the validation set
    '''

    X_valid = np.empty((0, 32 * 32 * 3))
    y_valid = np.empty(1)

    fname = '../data/data_batch_4'
    data_dict = unpack_data(fname)

    X_valid = data_dict['data']
    y_valid = data_dict['labels']

    X_valid = X_valid.reshape(
        (X_valid.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)

    return (X_valid, y_valid)


def load_testing():
    '''
    Loads and reshapes the data for the test set
    '''

    X_test = np.empty((0, 32 * 32 * 3))
    y_test = np.empty(1)

    fname = '../data/test_batch'
    data_dict = unpack_data(fname)

    X_test = data_dict['data']
    y_test = data_dict['labels']

    X_test = X_test.reshape(
        (X_test.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)

    return (X_test, y_test)


if __name__ == '__main__':

    X_train, y_train = load_training()
    X_valid, y_valid = load_validation()
    X_test, y_test = load_testing()
