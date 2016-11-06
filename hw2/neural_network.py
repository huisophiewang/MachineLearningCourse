import os
import scipy.io
import numpy as np

def f(x):
    return 1.0/(1+np.exp(-x))

def f_prime(x):
    return f(x)*(1-f(x))

def cost(x, y, w1, w2, b1, b2, lam=1):
    N = len(x)
    Jwb = 0
    for i in range(N):
        a1 = x[i].T
        z2 = np.dot(w1, a1)+b1
        a2 = f(z2)
        z3 = np.dot(w2, a2)+b2
        hi = f(z3)    
        Jwb += (hi-y[i])*(hi-y[i])/2
    Jwb = Jwb/N + lam/2.0*(np.sum(np.square(w1))+np.sum(np.square(w2)))
    print Jwb
    
def toy(x, y, alpha=1, lam=1):

    N = len(X)
    np.random.seed(0)
    
    ### initialization
    w1 = np.random.normal(0, 0.01, (3,4))
    w2 = np.random.normal(0, 0.01, (2,3))
    b1 = np.zeros(3)
    b2 = np.zeros(2)
    
    for iter in range(10):
        delta_w1 = np.zeros((3,4))
        delta_w2 = np.zeros((2,3))
        delta_b1 = np.zeros(3)
        delta_b2 = np.zeros(2)
        
        for i in range(N):
            # forward
            a1 = x[i].T
            z2 = np.dot(w1, a1)+b1
            a2 = f(z2)
            z3 = np.dot(w2, a2)+b2
            a3 = f(z3)

            # backward
            print '----------'
            error3 = (a3 - y[i])*f_prime(z3)
            delta_w2 += np.dot(np.array([error3]).T, [a2])
            delta_b2 += error3
    
            ### e.g. s2=3, s3=2
            #error20 = error3[0]*w2[0,0]*f_prime(z2)[0] + error3[1]*w2[1,0]*f_prime(z2)[0]
            #error21 = error3[0]*w2[0,1]*f_prime(z2)[1] + error3[1]*w2[1,1]*f_prime(z2)[1]
            #error22 = error3[0]*w2[0,2]*f_prime(z2)[2] + error3[1]*w2[1,2]*f_prime(z2)[2]
    
            error2 = (error3*w2.T).sum(axis=1)*f_prime(z2)
            delta_w1 += np.dot(np.array([error2]).T, [a1])
            delta_b1 += error2
            
        w1 = w1 - alpha*(delta_w1/N + lam*w1)
        w2 = w2 - alpha*(delta_w2/N + lam*w2)
        b1 = b1 - alpha*(delta_b1/N)
        b2 = b2 - alpha*(delta_b2/N)
        
        print w1
        print w2


        
        
    
#         # multiply how much we missed by the 
#         # slope of the sigmoid at the values in l1
#         l2_delta = l2_error * sigmoid(l1,True)
#     
#         # update weights
#         w0 += np.dot(l1.T,l1_delta)
    


if __name__ == '__main__':
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     data = scipy.io.loadmat(os.path.join(cur_dir, 'data.mat'))
#     x_train, y_train = data['X_trn'], data['Y_trn']
#     x_test, y_test = data['X_tst'], data['Y_tst']
    #print x_test
    X = np.array([[0,0,1,1],
                  [0,1,0,1]])           
    Y = np.array([[0,1],
                  [1,0]])
    toy(X, Y)
