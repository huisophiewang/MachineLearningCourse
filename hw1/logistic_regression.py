import scipy.io
import numpy as np
import os

def get_xi_class_prob(xi, theta, num_class):
    p = np.zeros((num_class, 1)) # probability of P(y=j|xi)
    for k in range(num_class):
        p[k] = np.exp(np.dot(theta[k].T, xi))
    p /= sum(p)  # normalize
    return p

def get_accuracy(x, y, theta):
    num_class = len(set(y.T[0]))
    n = len(x)
    x = np.concatenate((np.ones((n, 1)), x), axis=1)
    correct = 0
    for i in range(n):
        p = get_xi_class_prob(x[i], theta, num_class)
        predict = np.argmax(p)
        correct += int(predict==y[i])
    return float(correct)/n
    
def gradient_ascent(x_train, y_train, alpha=1):  
    N = x_train.shape[0]  # number of sample
    x_train = np.concatenate((np.ones((N, 1)), x_train), axis=1)
    M = x_train.shape[1]  # number of features
    K = len(set(y_train.T[0]))  # number of classes
    theta = np.zeros((K, M))  
    delta = np.zeros((K, M))
    
    prev_lik = float("-inf")
    while True:
        # update beta
        for i in range(N):
            p = get_xi_class_prob(x_train[i], theta, K)
            for j in range(K):   
                indicator = int(y_train[i][0] == j)  # 1 if yi=j
                delta[j] += x_train[i]*(indicator-p[j])  
        delta /= N 
        theta += alpha*delta
        
        # compute log likelihood
        lik=0.0
        for i in range(N):
            yi = (np.arange(K).reshape((K, 1)) == y_train[i]).astype(int) # convert yi to [1 0 0] 
            p = get_xi_class_prob(x_train[i], theta, K)
            lik += np.log(sum(p*yi)[0])
        print "log likelihood: %f" % lik
        if np.abs(lik - prev_lik) < 0.001:
            break    
        prev_lik = lik
    return theta

def softmax_cls(x_train, y_train, x_test, y_test):
    theta = gradient_ascent(x_train, y_train)
    print "theta:\n%s" % theta
    train_acc = get_accuracy(x_train, y_train, theta)
    print "training accuracy: %f" % train_acc
    test_acc = get_accuracy(x_test, y_test, theta)
    print "test accuracy: %f" % test_acc

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = scipy.io.loadmat(os.path.join(cur_dir, 'logistic_regression.mat'))
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst']
    
    print "=========================== Logistic Regression ============================="  
    softmax_cls(x_train, y_train, x_test, y_test)


    

