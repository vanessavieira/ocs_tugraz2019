import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt


if __name__ == '__main__':
    # load the data set
    data_set = np.load('./data.npz')
    # get the training data
    x_train = data_set['x_train']
    y_train = data_set['y_train']
    # get the test data
    x_test = data_set['x_test']
    y_test = data_set['y_test']

    print('Training data:', x_train.shape, y_train.shape)
    print('Test data:', x_test.shape, y_test.shape)