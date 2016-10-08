import scipy.io
import numpy as np

# transform matrix X using polynomial basis [x, x^2, ...]
# each column Xj is transformed to several columns [Xj, Xj^2, ...]
def poly_basis_trans(X, degree):
    num_col = X.shape[1]
    num_row = X.shape[0]

    X_trans = np.zeros((num_row, num_col*(degree+1)))
    for j in range(num_col):
        for d in range(degree+1):
            X_trans[:,j*(degree+1)+d] = np.power(X[:,j], d)
    return X_trans

        

def linear_reg(data):
    degree_cases = [2, 5, 10, 20]
    for degree in degree_cases:
        # reset x_train, x_test, since poly_basis_trans overwrites it
        x_train, y_train = data['X_trn'], data['Y_trn']
        x_test, y_test = data['X_tst'], data['Y_tst']
        print '----------------------'
        print "degree=%d" % degree
        x_train = poly_basis_trans(x_train, degree)
        pseudo_inv = np.linalg.inv(np.dot(np.transpose(x_train), x_train))
        pseduo_inv = np.dot(pseudo_inv, np.transpose(x_train))
        w = np.dot(pseduo_inv, y_train)
        print "Cofficients w: %s" % np.transpose(w)   
        train_err = np.mean((np.dot(x_train, w) - y_train) ** 2)
        print "Training error: %s" % train_err
        x_test = poly_basis_trans(x_test, degree)
        test_err = np.mean((np.dot(x_test, w) - y_test) ** 2)
        print "Test error: %s" % test_err

def get_w(x, y, lam):
    pseudo_inv = np.linalg.inv(np.dot(np.transpose(x), x) + lam * np.identity(x.shape[1]))
    pseduo_inv = np.dot(pseudo_inv, np.transpose(x))
    w = np.dot(pseduo_inv, y)
    print w
    
def get_holdout_err(x, y, x_hold, y_hold, lam):  

    # x should be standardlized!!!
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

def normalize_col(arr):
    for j in range(arr.shape[1]):
        cmean = np.mean(arr[:,j])
        cstd = np.std(arr[:,j])
        print cstd
        arr[:,j] = (arr[:,j]-cmean)/float(cstd)
    return arr
    
    
def ridge_reg(data):

    degree_cases = [2, 5, 10, 20]
    degree = 2
    
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst'] 
    x_train = poly_basis_trans(x_train, degree)
    x_test = poly_basis_trans(x_test, degree)
    
    
    
    
    # for fold K = [2, 5, 10, N]
    fold_cases = [2, 5, 10, len(x_train)]
    #for fold in fold_cases:
    fold = 10
    print '------------------'
    print "%d fold" % fold
    size = int(x_train.shape[0]/fold)
    
    # shuffle training data
    #p = np.random.permutation(len(x_train))
    #x_train = x_train[p]
    #y_train = y_train[p]

    # compute lambda for each fold and average them
    lams = []
    for i in range(fold):
        if i != 5:
            continue
        print 'fold: %d' % i
        idx_start = i*size
        idx_end = (i+1)*size
        
        #x_train = normalize_col(x_train[:,1:])
        #y_train = y_train - np.mean(y_train)
        
#         x_train_hold = x_train[idx_start:idx_end,:]
#         y_train_hold = y_train[idx_start:idx_end,:]
        x_train_train = np.concatenate((x_train[0:idx_start,:], x_train[idx_end:,:]), axis=0)
        y_train_train = np.concatenate((y_train[0:idx_start,:], y_train[idx_end:,:]), axis=0)
        
        x_train_train = normalize_col(x_train_train[:,1:])
        y_train_train = y_train_train - np.mean(y_train_train)

        #start_range = [10 ** j for j in range(-5, 6)]
        lam = 100
        get_w(x_train_train, y_train_train, lam)
        
#         lam = narrow_lambda_range(x_train_train, y_train_train, x_train_hold, y_train_hold, start_range, 10000)
#         print "lambda: %f" % lam
#         lams.append(lam)
    #best_lam = np.mean(lams)
    #print "RESULT: best lambda is %f" % best_lam 
    
    

    
def cross_validation(data, fold):
    np.random.shuffle(data)
    
    
if __name__ == '__main__':
    data = scipy.io.loadmat('linear_regression.mat')



    #print [10 ** i for i in range(-10, 10)]
    #print np.arange(0.1, 10, 0.05)
    
    #linear_reg(data)
    ridge_reg(data)

    
    


    
