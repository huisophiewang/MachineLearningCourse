from sklearn import linear_model
import scipy.io
import numpy as np

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

def ridge(x_tt, y_tt, x_hd, y_hd):
    for lam in [10 ** j for j in range(-5, 6)]:
    #for lam in [1]:
        print "lambda is %f" % lam
        #clf = linear_model.Ridge(alpha=alpha, fit_intercept=False)
        clf = linear_model.Ridge(alpha=lam)
    
        clf.fit(x_tt, y_tt)
        print('Coefficients: \n', clf.coef_)
        print clf.intercept_
        #print clf.predict(x_hd)
        mse = np.mean((clf.predict(x_hd) - y_hd) ** 2)

        print("Mean squared error: %f" % mse)
        
def ridge_normalize(x_tt, y_tt, x_hd, y_hd):
    #normalize x, center y
    x_tt_norm, means, stds = normalize_col(x_tt[:,1:])
    y_tt_center = y_tt - np.mean(y_tt)
    for lam in [10 ** j for j in range(-5, 6)]:
    #for lam in [1]:
        print "lambda is %f" % lam
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
def ridge_test(x_tt, y_tt, x_hd, y_hd):
    #normalize x, center y
    x_tt_center = x_tt - np.mean(x_tt, axis=0)
    y_tt_center = y_tt - np.mean(y_tt)

    for lam in [10 ** j for j in range(-5, 6)]:
    #for lam in [1]:
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

        x_hd_center = x_hd - np.mean(x_tt, axis=0)
        y_predict = clf.predict(x_hd_center[:,1:]) + np.mean(y_tt)
        #print y_predict
        mse = np.mean((y_predict - y_hd) ** 2)
        print("Mean squared error: %f" % mse)
        
        
                
def test():
    data = scipy.io.loadmat('linear_regression.mat')
    degree = 2
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst'] 
    x_train = poly_basis_trans(x_train, degree)
    x_test = poly_basis_trans(x_test, degree)
    
    # 10 fold, the 5th fold
    idx_start = 5*8
    idx_end = 6*8
    x_hd = x_train[idx_start:idx_end,:]
    y_hd = y_train[idx_start:idx_end,:]
    x_tt = np.concatenate((x_train[0:idx_start,:], x_train[idx_end:,:]), axis=0)
    y_tt = np.concatenate((y_train[0:idx_start,:], y_train[idx_end:,:]), axis=0)
    print x_hd
    print y_hd
    #ridge(x_tt, y_tt, x_hd, y_hd)
    #ridge_normalize(x_tt, y_tt, x_hd, y_hd)
    ridge_test(x_tt, y_tt, x_hd, y_hd)
    



    
    


if __name__ == '__main__':
    test()