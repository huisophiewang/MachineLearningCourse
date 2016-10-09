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


def get_ridge_w(x, y, lam):
    # center x, don't need to center y though
    # w is the same whether y is centered or not, see proof hw1 q6
    # w0 is not the same as in q6, since here we want w0 for the original x
    x_center = x[:,1:] - np.mean(x[:,1:], axis=0)
    pseudo_inv = np.linalg.inv(np.dot(np.transpose(x_center), x_center) + lam * np.identity(x_center.shape[1]))
    pseduo_inv = np.dot(pseudo_inv, np.transpose(x_center))
    w = np.dot(pseduo_inv, y)
    w0 = np.mean(y) - np.dot(np.mean(x[:,1:], axis=0), w)
    w0 = np.reshape(w0, (1,1))
    all_w = np.concatenate((w0, w), axis=0)
    return all_w

def linear_reg(x_train, y_train, x_test, y_test, degree):
    x_train = poly_basis_trans(x_train, degree)
    pseudo_inv = np.linalg.inv(np.dot(np.transpose(x_train), x_train))
    pseduo_inv = np.dot(pseudo_inv, np.transpose(x_train))
    w = np.dot(pseduo_inv, y_train)
    print "cofficients w: %s" % w.T 
    train_err = np.mean((np.dot(x_train, w) - y_train) ** 2)
    print "training mse: %s" % train_err
    x_test = poly_basis_trans(x_test, degree)
    test_err = np.mean((np.dot(x_test, w) - y_test) ** 2)
    print "test mse: %s" % test_err
       
def ridge_reg(x_train, y_train, x_test, y_test, degree, fold):
    x_train = poly_basis_trans(x_train, degree)
    x_test = poly_basis_trans(x_test, degree)
    lam_range = [10 ** j for j in range(-5, 6)]
    #lam_range = [1.0]
    lam_hd_err = []
    for lam in lam_range:
        #print 'lambda: %f' % lam
        fold_hd_err = []
        for k in range(fold):
            #print 'fold: %d' % k
            hd_idx = np.arange(k, len(x_train), fold)
            x_hd, y_hd = x_train[hd_idx], y_train[hd_idx]
            x_tt, y_tt = np.delete(x_train, hd_idx, axis=0), np.delete(y_train, hd_idx, axis=0)
            all_w = get_ridge_w(x_tt, y_tt, lam)
            #print all_w
            hd_err = np.mean((np.dot(x_hd, all_w) - y_hd) ** 2)
            #print hd_err
            fold_hd_err.append(hd_err)
        avg_hd_err = np.mean(fold_hd_err)
        lam_hd_err.append(avg_hd_err)
    #pprint(lam_hd_err)
    idx = np.argmin(lam_hd_err)
    lam_hat = lam_range[idx]
    print "best lambda: %f" % lam_hat
    w_hat = get_ridge_w(x_train, y_train, lam_hat)
    print "coefficients w: %s" % w_hat.T
    train_mse = np.mean((np.dot(x_train, w_hat) - y_train) ** 2)
    print "training mse: %f" % train_mse
    test_mse = np.mean((np.dot(x_test, w_hat) - y_test) ** 2)
    print "test mse: %f" % test_mse


if __name__ == '__main__':
    data = scipy.io.loadmat('linear_regression.mat')
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst']
    
    print "=========================== Linear Regression ============================="  
    for degree in [2, 5, 10, 20]:          
        print "--------- degree=%d ---------" % degree
        linear_reg(x_train, y_train, x_test, y_test, degree)
    
    print "=========================== Ridge Regression ============================="  
    for degree in [2, 5, 10, 20]:
        for fold in [2, 5, 10, len(x_train)]:   
            print "--------- degree=%d fold=%d ---------" % (degree, fold)
            ridge_reg(x_train, y_train, x_test, y_test, degree, fold)

    
    


    
