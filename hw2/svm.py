import os
import random
import copy
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def convert_y(Y, pos_label):
    result = [1 if y==pos_label else -1 for y in Y]
    return np.transpose([result])
    

def svo(X, Y, C=1, tol=0.001, max_passes=5):
    N = len(X)
    alpha = np.zeros((N,1))
    b = 0
    passes = 0
    #while(passes < 1):
    
    num_changed_alpha = 0
    for i in range(1):
        Ei = np.sum(alpha*Y*np.dot(X, np.array([X[i]]).T)) + b - Y[i]
        if (Y[i]*Ei < -tol and alpha[i] < C) or (Y[i]*Ei > tol and alpha[i] > 0):
            j = random.choice(np.delete(range(N), i))
            j=7
            Ej = np.sum(alpha*Y*np.dot(X, np.array([X[j]]).T)) + b - Y[j]
            alpha_i_old = copy.copy(alpha[i])
            alpha_j_old = copy.copy(alpha[j])
            print alpha_j_old
            
            if Y[i] == Y[j]:
                L = max(0, alpha[i]+alpha[j]-C)
                H = min(C, alpha[i]+alpha[j])
            else:
                L = max(0, alpha[j]-alpha[i])
                H = min(C, C+alpha[j]-alpha[i])
            if L == H:
                continue
            eta = 2*sum(X[i]*X[j]) - sum(X[i]*X[i]) - sum(X[j]*X[j])
            if eta >= 0:
                continue
            ### update alpha j
            print Ei, Ej
            alpha[j] -= Y[j]*(Ei-Ej)/eta
            print alpha[j], alpha_j_old
            
            if alpha[j] > H:
                alpha[j] = H
            if alpha[j] < L:
                alpha[j] = L
            if (abs(alpha[j]-alpha_j_old) < 10**(-5)):
                continue
            
            ### update alpha i
            alpha[i] += Y[i]*Y[j]*(alpha_j_old-alpha[j])
            print alpha

            b1 = b - Ei - Y[i]*(alpha[i]-alpha_i_old)*sum(X[i]*X[i]) - Y[j]*(alpha[j]-alpha_j_old)*sum(X[i]*X[j])
            b2 = b - Ej - Y[i]*(alpha[i]-alpha_i_old)*sum(X[i]*X[j]) - Y[j]*(alpha[j]-alpha_j_old)*sum(X[j]*X[j])
            
            print b1
            print b2
            if alpha[i] < C and alpha[i] > 0:
                b = b1
            elif alpha[j] < C and alpha[j] > 0:
                b = b2
            else:
                b = (b1+b2)/2
                
            num_changed_alpha += 1
            
            print b
                
                
#         if (num_changed_alpha == 0):
#             passes += 1
#         else:
#             passes = 0
            
    

    
    
if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = scipy.io.loadmat(os.path.join(cur_dir, 'data.mat'))
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst']
    #print y_test[:14]
    #print x_test[:14]
#     plt.plot(x_test[:7,0], x_test[:7,1], 'ro')
#     plt.plot(x_test[7:14,0], x_test[7:14,1], 'bo')
#     plt.show()
    y_test = convert_y(y_test, 0)
    svo(x_test[:14], y_test[:14])
