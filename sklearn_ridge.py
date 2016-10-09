from sklearn import linear_model
import scipy.io
import numpy as np
from pprint import pprint

def poly_basis_trans(X, degree):
    num_col = X.shape[1]
    num_row = X.shape[0]
    X_trans = np.zeros((num_row, num_col*(degree+1)))
    for j in range(num_col):
        for d in range(degree+1):
            X_trans[:,j*(degree+1)+d] = np.power(X[:,j], d)
    return X_trans

def normalize_col(arr):
    means, stds = [], []
    result = np.zeros(arr.shape)
    for j in range(arr.shape[1]):
        cmean = np.mean(arr[:,j])
        means.append(cmean)
        cstd = np.std(arr[:,j])
        stds.append(cstd)
        result[:,j] = (arr[:,j]-cmean)/float(cstd)
    return result, means, stds

def cov(a, b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)

def ridge(x_tt, y_tt, x_hd, y_hd, lam):
    #clf = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    clf = linear_model.Ridge(alpha=lam)

    clf.fit(x_tt, y_tt)
    print('Coefficients: \n', clf.coef_)
    print clf.intercept_
    #print clf.predict(x_hd)
    mse = np.mean((clf.predict(x_hd) - y_hd) ** 2)

    print("Mean squared error: %f" % mse)
        
def ridge_normalize(x_tt, y_tt, x_hd, y_hd, lam):
    #normalize x, center y
    x_tt_norm, means, stds = normalize_col(x_tt[:,1:])
    y_tt_center = y_tt - np.mean(y_tt)

    #clf = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    clf = linear_model.Ridge(alpha=lam, fit_intercept=False)

    clf.fit(x_tt_norm, y_tt_center)
    print('Coefficients: \n', clf.coef_)
    print clf.intercept_
    x_hd_norm = np.zeros(x_hd.shape)
    x_hd_norm[:,1]=(x_hd[:,1]-means[0])/stds[0]
    x_hd_norm[:,2]=(x_hd[:,2]-means[1])/stds[1]
    #print x_hd_norm

    y_predict = clf.predict(x_hd_norm[:,1:]) + np.mean(y_tt)
    #print y_predict
    mse = np.mean((y_predict - y_hd) ** 2)
    print("Mean squared error: %f" % mse)
        
def ridge_test(x_tt, y_tt, x_hd, y_hd, lam):
    #normalize x, center y
    x_tt_center = x_tt - np.mean(x_tt, axis=0)
    #y_tt_center = y_tt - np.mean(y_tt)
    y_tt_center = y_tt

    print "lambda is %f" % lam
    #clf = linear_model.Ridge(alpha=alpha, fit_intercept=False)
    clf = linear_model.Ridge(alpha=lam, fit_intercept=False)

    clf.fit(x_tt_center[:,1:], y_tt_center)
    print('Coefficients: \n', clf.coef_)
    print clf.intercept_
    
    #print np.mean(x_tt[:,1:], axis=0)
    #print clf.coef_.T
    w0 = np.mean(y_tt) - np.dot(np.mean(x_tt[:,1:], axis=0), clf.coef_.T)
    print w0
    #print np.mean(y_tt)

    x_hd_center = x_hd - np.mean(x_tt, axis=0)
    y_predict = clf.predict(x_hd_center[:,1:]) + np.mean(y_tt)
    #print y_predict
    mse = np.mean((y_predict - y_hd) ** 2)
    print("Mean squared error: %f" % mse)
    return mse
        
        
                
def main():
    data = scipy.io.loadmat('linear_regression.mat')
    degree, fold = 2, 10
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst'] 
    x_train = poly_basis_trans(x_train, degree)
    x_test = poly_basis_trans(x_test, degree)
    
    lam_errs = []
    lam_ws = []
    lam_range = [10 ** j for j in range(-5, 6)]
    lam_range = [1.0]
    for lam in lam_range:
        fold_errs = []
        for k in range(fold):
            print 'fold: %d' % k
            hd_idx = np.arange(k, len(x_train), fold)
            x_hd, y_hd = x_train[hd_idx], y_train[hd_idx]
            x_tt, y_tt = np.delete(x_train, hd_idx, axis=0), np.delete(y_train, hd_idx, axis=0)
            #print x_hd
            #print y_hd
            #ridge(x_tt, y_tt, x_hd, y_hd,lam)
            #ridge_normalize(x_tt, y_tt, x_hd, y_hd, lam)
            mse = ridge_test(x_tt, y_tt, x_hd, y_hd, lam)
            fold_errs.append(mse)
        lam_errs.append(np.mean(fold_errs))
    pprint(lam_errs)
    idx = np.argmin(lam_errs)
    print idx
    best_lam = lam_range[idx]
    print best_lam
            
    



    
    


if __name__ == '__main__':
    main()