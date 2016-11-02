import os
import random
import copy
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def convert_y(Y, pos_label):
    result = [1 if y==pos_label else -1 for y in Y]
    return np.transpose([result])

def svo(X, Y, C=1, tol=0.0001, max_passes=2):
    N = len(X)
    alpha = np.zeros((N,1))
    b = 0
    passes = 0
    while(passes < max_passes):
        num_changed_alpha = 0
        for i in range(N):
            Ei = np.sum(alpha*Y*np.dot(X, np.array([X[i]]).T)) + b - Y[i]
            if (Y[i]*Ei < -tol and alpha[i] < C) or (Y[i]*Ei > tol and alpha[i] > 0):
                j = random.choice(np.delete(range(N), i))
                Ej = np.sum(alpha*Y*np.dot(X, np.array([X[j]]).T)) + b - Y[j]
                alpha_i_old = copy.copy(alpha[i])
                alpha_j_old = copy.copy(alpha[j])
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
                alpha[j] -= Y[j]*(Ei-Ej)/eta
                if alpha[j] > H:
                    alpha[j] = H
                if alpha[j] < L:
                    alpha[j] = L
                if (abs(alpha[j]-alpha_j_old) < 10**(-5)):
                    continue                
                ### update alpha i
                alpha[i] += Y[i]*Y[j]*(alpha_j_old-alpha[j])
                #print alpha
                ### update b
                b1 = b - Ei - Y[i]*(alpha[i]-alpha_i_old)*sum(X[i]*X[i]) - Y[j]*(alpha[j]-alpha_j_old)*sum(X[i]*X[j])
                b2 = b - Ej - Y[i]*(alpha[i]-alpha_i_old)*sum(X[i]*X[j]) - Y[j]*(alpha[j]-alpha_j_old)*sum(X[j]*X[j])
                if alpha[i] < C and alpha[i] > 0:
                    b = b1
                elif alpha[j] < C and alpha[j] > 0:
                    b = b2
                else:
                    b = (b1+b2)/2
                #print b  
                num_changed_alpha += 1     
        if (num_changed_alpha == 0):
            passes += 1
        else:
            passes = 0           
    return alpha, b
    
def margin(x_train, y_train, x_test, y_test, alpha, b):
    #np.sum(alpha*Y*np.dot(X, np.array([X[i]]).T)) + b
    N, K = x_train.shape
    w = np.zeros(K)
    for i in range(N):
        if alpha[i]!=0:
            w += alpha[i]*y_train[i]*x_train[i]
            
    f = np.dot(x_test, np.array([w]).T)+b
    #predict = np.sign(f)
    geo_margin = f/sum(w*w)
    return geo_margin

def accuracy(y_predict, y_test):   
    correct = sum(y_predict == y_test)
    acc = float(correct[0])/len(y_test)
    return acc
    
def svm_cls(x_train, y_train, x_test, y_test, C):
    labels = set(y_train.T[0])
    margins = np.zeros((len(x_test), len(labels)))

    for k, label in enumerate(labels):
        y_train_binary = convert_y(y_train, label)
        alpha, b = svo(x_train, y_train_binary, C)
        margins[:,k] = margin(x_train, y_train_binary, x_test, y_test, alpha, b).T
        
    y_predict = np.array([np.argmax(margins, axis=1)]).T
    #print y_predict
    acc = accuracy(y_predict, y_test)
    print "test accuracy: %f" % acc
    

def plot_test(x_test, w, b):      
    plt.plot(x_test[:7,0], x_test[:7,1], 'ro')
    plt.plot(x_test[7:14,0], x_test[7:14,1], 'go')
    plt.plot(x_test[14:,0], x_test[14:,1], 'bo')
#     x1 = np.linspace(-2, 2)
#     x2 = [(-b-w[0]*x)/w[1] for x in x1]
#     plt.plot(x1, x2)
    plt.show()
    
    
if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = scipy.io.loadmat(os.path.join(cur_dir, 'data.mat'))
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst']
    
    C_range = [10 ** j for j in range(-5, 3)]
    for C in C_range:
        print "set C = %f" % C
        svm_cls(x_train, y_train, x_test, y_test, C)

