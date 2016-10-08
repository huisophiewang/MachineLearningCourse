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
    for j in range(arr.shape[1]):
        cmean = np.mean(arr[:,j])
        cstd = np.std(arr[:,j])
        arr[:,j] = (arr[:,j]-cmean)/float(cstd)
    return arr

def ridge():
    data = scipy.io.loadmat('linear_regression.mat')
    degree = 2
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst'] 
    x_train = poly_basis_trans(x_train, degree)
    x_test = poly_basis_trans(x_test, degree)
    
    idx_start = 5*8
    idx_end = 6*8
    x_train_hold = x_train[idx_start:idx_end,:]
    y_train_hold = y_train[idx_start:idx_end,:]
    x_train_train = np.concatenate((x_train[0:idx_start,:], x_train[idx_end:,:]), axis=0)
    y_train_train = np.concatenate((y_train[0:idx_start,:], y_train[idx_end:,:]), axis=0)
    #print x_train_hold
    #print y_train_hold
    x_train_train = normalize_col(x_train_train[:,1:])
    y_train_train = y_train_train - np.mean(y_train_train)
    
    for alpha in [10 ** j for j in range(-5, 6)]:
        print "lambda is %f" % alpha
        clf = linear_model.Ridge(alpha=alpha, fit_intercept=False)

        clf.fit(x_train_train, y_train_train)
        print('Coefficients: \n', clf.coef_)
        print clf.intercept_
#         print clf.predict(x_train_hold)
#         mse = np.mean((clf.predict(x_train_hold) - y_train_hold) ** 2)
#         print("Mean squared error: %f" % mse)

if __name__ == '__main__':
    ridge()