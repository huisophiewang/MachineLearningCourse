import os
import scipy.io
import numpy as np

def f(x):
    return 1.0/(1.0+np.exp(-x))

def f_prime(x):
    return f(x)*(1.0-f(x))

def softmax(x):
    p = np.exp(x)
    p /= sum(p)  # normalize
    return p

def cost_func(x, y, w1, w2, b1, b2, lam=1):
    N = len(x)
    Jwb = 0
    for i in range(N):
        a1 = x[i].T
        z2 = np.dot(w1, a1)+b1
        a2 = f(z2)
        z3 = np.dot(w2, a2)+b2
        hi = f(z3)    
        Jwb += np.dot((hi-y[i]), (hi-y[i]))/2.0
    Jwb = Jwb/N + lam/2.0*(np.sum(np.square(w1))+np.sum(np.square(w2)))
    #Jwb = Jwb/N 
    print Jwb
    return Jwb
    
    
def convert_y(y):
    N = len(y)
    k = len(set(y.T[0]))
    result = np.zeros((N, k))
    for i in range(N):
        label = y[i][0]
        result[i][label] = 1
    return result
    
def neural_net(x, y, s2=3, alpha=10.0, lam=0.001):
    
    N, s1 = x.shape
    s3 = y.shape[1]

    ### initialize
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
        print '=========== iter %d ===========' % iter

        delta_w1 = np.zeros((s2,s1))
        delta_w2 = np.zeros((s3,s2))
        delta_b1 = np.zeros(s2)
        delta_b2 = np.zeros(s3)
        
        for i in range(N):
            print '------'
            # forward
            print x[i]
            a1 = x[i]
            z2 = np.dot(w1, a1)+b1
            a2 = f(z2)
            z3 = np.dot(w2, a2)+b2
            #print z3
            #a3 = f(z3)
            a3 = softmax(z3)
            print "predict:"
            print a3
            

            # backward
            error3 = (a3 - y[i])*f_prime(z3)
            print "errors:"
            print error3
            delta_w2 += np.dot(np.array([error3]).T, [a2])
            #print delta_w2
            delta_b2 += error3
    
            ### e.g. s2=3, s3=2
            #error20 = error3[0]*w2[0,0]*f_prime(z2)[0] + error3[1]*w2[1,0]*f_prime(z2)[0]
            #error21 = error3[0]*w2[0,1]*f_prime(z2)[1] + error3[1]*w2[1,1]*f_prime(z2)[1]
            #error22 = error3[0]*w2[0,2]*f_prime(z2)[2] + error3[1]*w2[1,2]*f_prime(z2)[2]
            # in matrix form:
            error2 = (error3*w2.T).sum(axis=1)*f_prime(z2)
            print error2
            delta_w1 += np.dot(np.array([error2]).T, [a1])
            delta_b1 += error2
 
        print 'gradient of w:'
        print delta_w1/N + lam*w1
        print delta_w2/N + lam*w2         
            
        w1 = w1 - alpha*(delta_w1/N + lam*w1)
        w2 = w2 - alpha*(delta_w2/N + lam*w2)
#         w1 = w1 - alpha*(delta_w1/N)
#         w2 = w2 - alpha*(delta_w2/N)
        b1 = b1 - alpha*(delta_b1/N)
        b2 = b2 - alpha*(delta_b2/N)
        
        print 'w:'
        print w1
        print w2
        print 'b:'
        print b1
        print b2
                
        #cost = np.log(cost_func(x, y, w1, w2, b1, b2, lam))
#         cost = cost_func(x, y, w1, w2, b1, b2, lam)
#         print "cost: %f" % cost
#         if np.abs(cost - prev_cost) < 10**(-6):
#             break
#         prev_cost = cost
    return w1, w2, b1, b2

def classify(x, y, w1, w2, b1, b2):
    N= len(x)
    for i in range(N):
        a1 = x[i].T
        z2 = np.dot(w1, a1)+b1
        a2 = f(z2)
        z3 = np.dot(w2, a2)+b2
        a3 = f(z3)
        print a3
        


def check_grad(x, y):
    w1 = np.array([[ 0.01764052,  0.00400157,  0.00978738],
                 [ 0.02240893,  0.01867558, -0.00977278],
                 [ 0.00950088, -0.00151357, -0.00103219],
                 [ 0.00410599,  0.00144044,  0.01454274],
                 [ 0.00761038,  0.00121675,  0.00443863]])
    w2 = np.array([[ 0.00333674,  0.01494079, -0.00205158,  0.00313068, -0.00854096],
                 [-0.0255299,   0.00653619,  0.00864436, -0.00742165,  0.02269755],
                 [-0.01454366,  0.00045759, -0.00187184,  0.01532779,  0.01469359]])
    b1 = np.zeros(5)
    b2 = np.zeros(3)
    
    #c = cost_func(x, y, w1, w2, b1, b2)
    
    epsilon = 10**(-4)
    w1[0][2] += epsilon
    print w1
    c1 = cost_func(x, y, w1, w2, b1, b2)
    w1[0][2] -= epsilon*2
    print w1
    c2 = cost_func(x, y, w1, w2, b1, b2)
    
    grad_approximate = (c1 - c2) / (2.0*epsilon)
    #print c1 - c2
    print grad_approximate
    
    


if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = scipy.io.loadmat(os.path.join(cur_dir, 'data.mat'))
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst']
    #print x_test


    y_test = convert_y(y_test)
    
    #w1, w2, b1, b2 = neural_net(x_test, y_test)
    #classify(x_test, y_test, w1, w2, b1, b2)
    #print '=============================='
    #check_grad(x_test, y_test)
   

    #x = np.identity(8)
    #y = np.identity(8)
    #y = np.array([[0],[0],[1],[1]])

    neural_net(x_test, y_test)

    
    
