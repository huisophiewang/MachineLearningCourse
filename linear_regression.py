import scipy.io
import numpy as np
from pprint import pprint

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

def get_ridge_w(x, y, lam):
    pseudo_inv = np.linalg.inv(np.dot(np.transpose(x), x) + lam * np.identity(x.shape[1]))
    pseduo_inv = np.dot(pseudo_inv, np.transpose(x))
    w = np.dot(pseduo_inv, y)
    return w
    
def ridge_reg(x_train, y_train, x_test, y_test, degree, fold):

    x_train = poly_basis_trans(x_train, degree)
    x_test = poly_basis_trans(x_test, degree)
    # shuffle
#         p = np.random.permutation(len(x_train))
#         x_train = x_train[p]
#         y_train = y_train[p]
    

    size = int(x_train.shape[0]/fold)

    lam_range = [10 ** j for j in range(-5, 6)]
    #lam_range = [1.0]
    lam_hd_err = []
    for lam in lam_range:
        #print 'lambda: %f' % lam
        fold_hd_err = []
        for k in range(fold):
            #print 'fold: %d' % k
            test_index = np.arange(k, len(x_train), fold)
            x_hd, y_hd = x_train[test_index], y_train[test_index]
            x_tt, y_tt = np.delete(x_train, test_index, axis=0), np.delete(y_train, test_index, axis=0)

            ### center x and y
            x_tt_center = x_tt[:,1:] - np.mean(x_tt[:,1:], axis=0)
            y_tt_center = y_tt - np.mean(y_tt)
            w = get_ridge_w(x_tt_center, y_tt_center, lam)
            x_hd_center = x_hd[:,1:] - np.mean(x_tt[:,1:], axis=0)
            print w 
            y_predict = np.dot(x_hd_center, w) + np.mean(y_tt)
            hd_err = np.mean((y_predict - y_hd) ** 2)
            fold_hd_err.append(hd_err)
            #print hd_err
        avg_hd_err = np.mean(fold_hd_err)
        #print "Average holdout error: %f" % avg_hd_err
        lam_hd_err.append(avg_hd_err)
    pprint(lam_hd_err)
    idx = np.argmin(lam_hd_err)
    print idx
    best_lam = lam_range[idx]
    print "best lambda is %f" % best_lam



    

    
    
if __name__ == '__main__':
    data = scipy.io.loadmat('linear_regression.mat')
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst']
    #print "================Linear Regression========================="             
    #linear_reg(x_train, y_train, x_test, y_test)
    
    #print "================Ridge Regression========================="  
#     for degree in [2, 5, 10, 20]:
#         for fold in [2, 5, 10, len(x_train)]:
        
    ridge_reg(x_train, y_train, x_test, y_test, degree=2, fold=10)

    
    


    
