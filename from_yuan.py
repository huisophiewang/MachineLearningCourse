__author__ = 'yuanzhong'

import numpy as np
from numpy.linalg import *
from optparse import OptionParser
import matplotlib.pyplot as plt
import scipy.io
import math
from pprint import pprint

def get_parameter():
    parser = OptionParser()
    parser.add_option("--i", dest="input", help="destination of the input data")
    parser.add_option("--p", dest="num_poly", help="number of dimensions")
    parser.add_option("--f", dest="fold", help="number of folds for cross-validation")
    return parser.parse_args()

def test(theta, data, mean):
    test_row = len(data)
    x = data[:, :-1]
    y = data[:, -1]

    sse = np.sum((np.sum(x*theta, axis=1)+mean-y)**2)
    rmse = np.sqrt(sse/test_row)
    return sse/test_row
    #print (np.sum(x*theta, axis=1)+mean-y)**2
    #return rmse, sse

def training(data, hyper_parameter):
    x = data[:, :-1]
    y = data[:, -1]
    x_t = x.transpose()
    # return np.dot(np.dot(pinv(np.dot(x_t, x)+hyper_parameter), x_t), y)
    return np.dot(np.dot(pinv(np.dot(x_t, x)+hyper_parameter*np.eye(x.shape[1])), x_t), y)

def cross_validation(data, degree, fold):
    row = len(data)
    size = int(row/fold)
    x = data[:, :-1]
    y = data[:, -1]
    feature = np.empty((row, 0), float)
    x_axis = list()
    rmse_train = list()
    rmse_test = list()

    for p in range(1, degree+1):
        feature = np.append(feature, x**p, axis=1)
    #print feature.shape
    #print y[:, np.newaxis].shape
    data_extend = np.concatenate((feature, y[:, np.newaxis]), axis=1)
    #print data_extend.shape
    #print data_extend[0, :]
    #data_extend = data_extend[np.random.permutation(row)]
    hyper_parameter = 0
    lam_range = [10 ** j for j in range(-5, 6)]
    #lam_range = [1.0]
    for hyper_parameter in lam_range:
        print hyper_parameter
        x_axis.append(math.log10(hyper_parameter))
        rmse_iter_train = np.zeros(fold)
        rmse_iter_test = np.zeros(fold)
        for i in range(fold):
            #print "=======================cross validation %d=========================================================" % (i)
            test_index = np.arange(i, row, fold)
            test_data = data_extend[test_index]
            train_data = np.delete(data_extend, test_index, axis=0)
            
#             idx_start = i*size
#             idx_end = (i+1)*size
#             x_hd = data_extend[idx_start:idx_end,:-1]
#             y_hd = data_extend[idx_start:idx_end,-1:]
#             x_tt = np.concatenate((data_extend[0:idx_start,:-1], data_extend[idx_end:,:-1]), axis=0)
#             y_tt = np.concatenate((data_extend[0:idx_start,-1:], data_extend[idx_end:,-1:]), axis=0)
#             train_data = np.concatenate((x_tt, y_tt), axis=1)
#             test_data = np.concatenate((x_hd, y_hd), axis=1)

            # mean = np.mean(train_data[:, -1])
            train_mean = np.mean(train_data, axis=0)
            # train_data = train_data-np.mean(train_data, axis=0)
            train_data = train_data - train_mean
            theta = training(train_data, hyper_parameter)
            print theta
            #rmse_iter_train[i] = test(theta, train_data, 0)[0]

            # test_data[:, :-1] = test_data[:, :-1]-np.mean(test_data[:, :-1], axis=0)
            # print train_mean[:-1].shape
            # print test_data[:, :-1].shape
            test_data[:, :-1] = test_data[:, :-1]-train_mean[:-1]
            #print test(theta, test_data, train_mean[-1])[1]/len(test_data)
            rmse_iter_test[i] = test(theta, test_data, train_mean[-1])
        rmse_train.append(np.mean(rmse_iter_train))
        rmse_test.append(np.mean(rmse_iter_test))
        #hyper_parameter += 0.2
    pprint(rmse_test)
    idx = np.argmin(rmse_test)
    print idx
    print lam_range[idx]
#     plt.title('Training set RMSE vs lambda with power='+str(degree))
#     plt.xlabel(r'$\lambda$')
#     plt.ylabel('RMSE')
#     plt.plot(x_axis, rmse_train, color = 'red', label="train set")
#     print rmse_train
#     plt.show()

#     plt.title('Test set RMSE vs lambda with power='+str(degree))
#     plt.xlabel(r'$\lambda$')
#     plt.ylabel('RMSE')
#     plt.plot(x_axis, rmse_test, color = 'blue', label="test set")
#     pprint(rmse_test)
#     idx = np.argmin(rmse_test)
#     print idx
#     print lam_range[idx]
#     plt.show()



if __name__ == "__main__":
    input = scipy.io.loadmat('linear_regression.mat')
    x_train, y_train = input['X_trn'], input['Y_trn']
    #print x_train
    #print y_train.shape
    data = np.concatenate((x_train, y_train), axis=1)

    cross_validation(data, degree=5, fold=2)