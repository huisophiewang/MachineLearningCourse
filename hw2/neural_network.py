import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

    
def toy():
    # input dataset
    X = np.array([  [0,0,1],
                    [0,1,1],
                    [1,0,1],
                    [1,1,1] ])
        
    # output dataset            
    y = np.array([[0,0,1,1]]).T
    
    # seed random numbers to make calculation
    # deterministic (just a good practice)
    np.random.seed(1)
    
    # initialize weights randomly with mean 0
    syn0 = 2*np.random.random((3,1)) - 1
    
    for iter in xrange(10000):
    
        # forward propagation
        l0 = X
        l1 = sigmoid(np.dot(l0,syn0))
    
        # how much did we miss?
        l1_error = y - l1
    
        # multiply how much we missed by the 
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * sigmoid(l1,True)
    
        # update weights
        syn0 += np.dot(l0.T,l1_delta)
    
    print "Output After Training:"
    print l1

if __name__ == '__main__':
    np.random.seed(0)
    syn0 = 2*np.random.random((3,1)) - 1
    print np.random.random((3,1))
    print np.random.normal(0, 0.01, (3,1))
