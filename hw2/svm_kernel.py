import os
import scipy.io
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def svm(kernel_name, C):
    clf = SVC(kernel=kernel_name, C=C)
    clf.fit(x_train, np.ravel(y_train))
    y_predict = clf.predict(x_test)
    acc = accuracy_score(np.ravel(y_test), y_predict)
    print "  accuracy: %f" % acc
    
if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = scipy.io.loadmat(os.path.join(cur_dir, 'data.mat'))
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst']
    
    kernels = ['linear','rbf', 'poly']
    C_range = [10 ** j for j in range(-5, 5)]
    for kernel in kernels:
        print 'kernel: %s' % kernel
        for C in C_range:
            print "  C=%f" % C
            svm(kernel, C)
