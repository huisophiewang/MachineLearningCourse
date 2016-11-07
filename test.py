import numpy as np

def double_shuffle(x_train, y_train):
    #shuffle x and y together
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]
    
def get_holdout_err(x, y, x_hold, y_hold, lam):  

    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x) + lam * np.identity(x.shape[1])), np.transpose(x)), y)
    print lam
    print w
    predict = np.dot(x_hold, w)
    print predict
    holdout_err = np.mean((np.dot(x_hold, w) - y_hold) ** 2)   
    return holdout_err

def narrow_lambda_range(x_tt, y_tt, x_hd, y_hd, lam_range, prev_min_err):
    print lam_range
    errs = []
    for lam in lam_range:
        holdout_err = get_holdout_err(x_tt, y_tt, x_hd, y_hd, lam)
        errs.append(holdout_err)
    print errs
    min_err = min(errs)
    idx = np.argmin(errs)
    best_lam = lam_range[idx]
    #print "lambad: %f, min err: %.3f" % (best_lam, min_err)
    err_diff = abs(min_err - prev_min_err)
    if err_diff < 0.01:
        return best_lam
    else:
        if idx==0 or idx==len(lam_range)-1:
            return best_lam
        else:
            new_lam_range = np.linspace(lam_range[idx-1], lam_range[idx+1], 100)
            return narrow_lambda_range(x_tt, y_tt, x_hd, y_hd, new_lam_range, min_err)
        

def standalize_col(arr):
    for j in range(arr.shape[1]):
        print j
        cmean = np.mean(arr[:,j])
        print cmean
        cstd = np.std(arr[:,j])
        print cstd
        arr[:,j] = (arr[:,j]-cmean)/float(cstd)
    return arr

if __name__ == '__main__':
    X1 = np.array([[1,1,2,2], 
                   [3,3,4,4]])
    X2 = np.array([[1], 
                  [11]])
    X3 = np.array([[1,2], 
                   [3,4],
                   [5,6]])
    X4 = np.array([1,2])
    X5 = np.array([[1,2,2]])
    X6 = np.array([[2], 
                  [1]])
    
    #print X4*X3
    print X5[0].T
    print X1.shape
    print X5[0].T.shape
    print np.dot(X1, X5[0])
#     print np.square(X3)
#     print X2*X6
#     print np.sum(X6)
#     print np.zeros((3, 1))

#     #print np.dot(X3, X2)
#     print X3.shape
#     print X4.shape
#     print X5.shape
#     
# 
#     print np.append(X3, X5, axis=0)
#     print np.concatenate((X3, X5), axis=0)
#     print np.insert(X3, 0, X5, axis=0)
    np.random.seed(0)
    #print np.random.permutation(5)
    #print np.linspace(-2, 2)
    

    
    
    


