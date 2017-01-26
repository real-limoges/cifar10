'''
This module opens the CIFAR10 data, transforms it into
numpy arrays, and then builds and trains a CNN.

'''
import cPickle
import glob
import numpy as np




def unpack_data(filename):
    '''
    Unpacks the file named filename as a python dictionary.
    '''
    reader = open(filename, 'rb')
    data_dict = cPickle.load(reader)
    reader.close()

    return data_dict

def build_model():
    pass    

if __name__ == '__main__':
    training_data = glob.glob('../data/data_batch_*')
    
    X_train = np.empty((0,32*32*3))
    y_train = np.empty(1)

    for i in xrange(1,6):
        fname = '../data/data_batch_{}'.format(i)
        data_dict = unpack_data(fname)

        if i == 1:
            X_train = data_dict['data']
            y_train = data_dict['labels']
        else:
            X_train = np.vstack((X_train, data_dict['data']))
            y_train = np.hstack((y_train, data_dict['labels']))
   
    foo = X_train.reshape(X_train.shape[0], 3, 32, 32)
    from PIL import Image

    img = Image.fromarray(foo[0], 'RGB')
    img.show()
