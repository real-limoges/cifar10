'''
This module opens the CIFAR10 data, transforms it into
numpy arrays, and then builds and trains a CNN.

'''
import cPickle
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, \
    Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.metrics import precision, recall, categorical_accuracy

# Number of categories/classes
NB_CLASSES = 10


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
        (stacked_features.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32') / 255.
    training_labels = np_utils.to_categorical(stacked_labels, NB_CLASSES)

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
        (X.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32') / 255.
    y_v = np_utils.to_categorical(y, NB_CLASSES)

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
        (X.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32') / 255.
    y_t = np_utils.to_categorical(y, NB_CLASSES)

    return (X_t, y_t)


def build_model(X):
    '''
    Creates a conv net model for the CIFAR10 data
    '''
    cifar = Sequential()

    cifar.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X.shape[1:]))
    cifar.add(Activation('relu'))
    cifar.add(Convolution2D(32, 3, 3))
    cifar.add(MaxPooling2D(pool_size=(2, 2)))
    cifar.add(Dropout(0.25))

    cifar.add(Convolution2D(64, 3, 3, border_mode='same'))
    cifar.add(Activation('relu'))
    cifar.add(Convolution2D(64, 3, 3))
    cifar.add(Activation('relu'))
    cifar.add(MaxPooling2D(pool_size=(2, 2)))
    cifar.add(Dropout(0.25))

    cifar.add(Flatten())
    cifar.add(Dense(512))
    cifar.add(Activation('relu'))
    cifar.add(Dropout(0.25))
    cifar.add(Dense(NB_CLASSES))
    cifar.add(Activation('softmax'))

    cifar.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', 'precision', 'recall'])

    return cifar


if __name__ == '__main__':

    features_tr, labels_tr = load_training()
    features_v, labels_v = load_validation()
    features_te, labels_te = load_testing()

    train = True 

    if train:
      cifar_model = build_model(features_tr)

      cifar_model.fit(features_tr, labels_tr,
                    batch_size=64,
                    nb_epoch=20,
                    validation_data=(features_v, labels_v),
                    shuffle=True)
      cifar_model.save('../data/cifar_model_2.h5')
    
    else:
      cifar_model = load_model('../data/cifar_model.h5')

      predictions = cifar_model.evaluate(features_te, labels_te)

      print cifar_model.metrics_names

