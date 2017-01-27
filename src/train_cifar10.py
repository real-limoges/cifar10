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
    for i in xrange(1, 5):
        fname = '../data/data_batch_{}'.format(i)
        data_dict = unpack_data(fname)

        if i == 1:
            stacked_features = data_dict['data']
            stacked_labels = data_dict['labels']
        else:
            stacked_features = np.vstack((stacked_features, 
                data_dict['data']))
            stacked_labels = np.hstack((stacked_labels, 
                data_dict['labels']))

    training_features = stacked_features.reshape(
        (stacked_features.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
    training_labels = stacked_labels

    return (training_features, training_labels)


def load_validation():
    '''
    Loads and reshapes the data for the validation set
    '''
    fname = '../data/data_batch_4'
    data_dict = unpack_data(fname)

    X = data_dict['data']
    y = data_dict['labels']

    X_v = X.reshape(
        (X.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
    y_v = y

    return (X_v, y_v)


def load_testing():
    '''
    Loads and reshapes the data for the test set
    '''
    fname = '../data/test_batch'
    data_dict = unpack_data(fname)

    X = data_dict['data']
    y = data_dict['labels']

    X_t = X.reshape(
        (X.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
    y_t = y

    return (X_t, y_t)


if __name__ == '__main__':

    features_tr, labels_tr = load_training()
    features_v, labels_v = load_validation()
    features_te, labels_te = load_testing()
