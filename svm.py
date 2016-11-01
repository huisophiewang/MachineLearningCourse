import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt

def convert_y(Y, pos_label):
    result = [1 if y==pos_label else -1 for y in Y]
    return np.transpose([result])
    

def svo(X, Y):
    N = len(X)
    alpha = np.zeros((N,1))
    b = 0
    i = 0
    e = np.sum(alpha*Y*np.dot(X, X[i])) + b - Y[i]

    
    
if __name__ == '__main__':
    data = scipy.io.loadmat('data.mat')
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst']
    #print y_test[:14]
    #print x_test[:14]
#     plt.plot(x_test[:7,0], x_test[:7,1], 'ro')
#     plt.plot(x_test[7:14,0], x_test[7:14,1], 'bo')
#     plt.show()
    y_test = convert_y(y_test, 0)
    svo(x_test[:14], y_test[:14])
