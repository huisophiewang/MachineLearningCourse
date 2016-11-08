import os
import scipy.io
import numpy as np

def f(x):
    return 1.0/(1.0+np.exp(-x))

def f_prime(x):
    return f(x)*(1.0-f(x))

def softmax(x):
    p = np.exp(x)
    p /= np.sum(p)  
    return p

def cost_func(x, y, w1, w2, b1, b2, lam=1):
    N = len(x)
    Jwb = 0.0
    for i in range(N):
        a1 = x[i].T
        z2 = np.dot(w1, a1)+b1
        a2 = f(z2)
        z3 = np.dot(w2, a2)+b2
        hi = f(z3)    
        Jwb += np.dot((hi-y[i]), (hi-y[i]))/2.0
    Jwb = Jwb/N + lam/2.0*(np.sum(np.square(w1))+np.sum(np.square(w2)))
    return Jwb
    
def convert_y(y):
    N = len(y)
    k = len(set(y.T[0]))
    result = np.zeros((N, k))
    for i in range(N):
        label = y[i][0]
        result[i][label] = 1
    return result

def nn(x, y, s2, alpha=10.0, lam=0.001):
    
    N, s1 = x.shape
    s3 = y.shape[1]

    ### initialize weights
    np.random.seed(0)
    w1 = np.random.normal(0, 0.01, (s2,s1))
    w2 = np.random.normal(0, 0.01, (s3,s2))
    b1 = np.zeros(s2)
    b2 = np.zeros(s3)
    prev_cost = float("inf")
#     w1 = np.array([[0.01, 0.01, 0.01, 0.01],
#                    [0.01, 0.01, 0.01, 0.01]])
#     w2 = np.array([[0.01, 0.01]])

    
    for iter in range(1000):
        #print '=========== iter %d ===========' % iter

        delta_w1 = np.zeros((s2,s1))
        delta_w2 = np.zeros((s3,s2))
        delta_b1 = np.zeros(s2)
        delta_b2 = np.zeros(s3)
        
        for i in range(N):
            # forward propagation
            a1 = x[i]
            z2 = np.dot(w1, a1)+b1
            a2 = f(z2)
            z3 = np.dot(w2, a2)+b2
            #a3 = f(z3)
            a3 = softmax(z3)
            
#             if (iter%100==0):
#                 print a3
                
            # backward propagation
            error3 = (a3 - y[i])*f_prime(z3)
            delta_w2 += np.dot(np.array([error3]).T, [a2])
            delta_b2 += error3
    
            ### e.g. s2=3, s3=2
            #error20 = error3[0]*w2[0,0]*f_prime(z2)[0] + error3[1]*w2[1,0]*f_prime(z2)[0]
            #error21 = error3[0]*w2[0,1]*f_prime(z2)[1] + error3[1]*w2[1,1]*f_prime(z2)[1]
            #error22 = error3[0]*w2[0,2]*f_prime(z2)[2] + error3[1]*w2[1,2]*f_prime(z2)[2]
            # in matrix form:
            error2 = (error3*w2.T).sum(axis=1)*f_prime(z2)
            delta_w1 += np.dot(np.array([error2]).T, [a1])
            delta_b1 += error2
 
#         print 'gradient of w:'
#         print delta_w1/N + lam*w1
#         print delta_w2/N + lam*w2         
            
        w1 = w1 - alpha*(delta_w1/N + lam*w1)
        w2 = w2 - alpha*(delta_w2/N + lam*w2)
        b1 = b1 - alpha*(delta_b1/N)
        b2 = b2 - alpha*(delta_b2/N)
        
#         print 'w:'
#         print w1
#         print w2
                
        #cost = np.log(cost_func(x, y, w1, w2, b1, b2, lam))
        cost = cost_func(x, y, w1, w2, b1, b2, lam)
        #print "cost: %.32f" % cost
        if np.abs(cost - prev_cost) < 10**(-5):
            break
        prev_cost = cost
    return w1, w2, b1, b2

def get_accuracy(x, y, w1, w2, b1, b2):
    N = len(x)
    correct = 0
    for i in range(N):
        a1 = x[i].T
        z2 = np.dot(w1, a1)+b1
        a2 = f(z2)
        z3 = np.dot(w2, a2)+b2
        a3 = f(z3)
        #print a3
        if np.argmax(a3) == np.argmax(y[i]):
            correct += 1
    return float(correct)/N
 
def check_grad(x, y, w1, w2, b1, b2):
    #c = cost_func(x, y, w1, w2, b1, b2)
    epsilon = 10**(-4)
    w1[0][2] += epsilon
    print w1
    c1 = cost_func(x, y, w1, w2, b1, b2)
    w1[0][2] -= epsilon*2
    print w1
    c2 = cost_func(x, y, w1, w2, b1, b2)
    grad_approximate = (c1 - c2) / (2.0*epsilon)
    print grad_approximate       

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = scipy.io.loadmat(os.path.join(cur_dir, 'data.mat'))
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst']

    y_train = convert_y(y_train)
    y_test = convert_y(y_test)
    
    # use autoencoder to test (4x2x4)
    #x = np.identity(4)
    #y = np.identity(4)

    for s2 in [10, 20, 30, 50, 100]:
        print "set s2 = %d" % s2
        w1, w2, b1, b2 = nn(x_train, y_train, s2=s2)
        acc = get_accuracy(x_test, y_test, w1, w2, b1, b2)
        print "test accuracy: %f" % acc

    
    
