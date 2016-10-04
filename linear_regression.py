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
    
def ridge_reg(data):

    degree_cases = [2, 5, 10, 20]
    degree = 2
    
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst'] 
    x_train = poly_basis_trans(x_train, degree)
    x_test = poly_basis_trans(x_test, degree)
    
    fold_cases = [2, 5, 10, x_train.shape[0]]
    fold = 2
    
    np.random.shuffle(x_train, fold)
    fold_size = int(x_train.shape[0]/fold)
    x_train_test, x_train_train = x_train[:fold_size,:], x_train[fold_size:,:]
    
    #for lam in np.arange(0.001, 1000):
    #for lam in [10 ** i for i in range(-3, 4)]:
        
    lam = 0.01
    
        

    
def cross_validation(data, fold):
    np.random.shuffle(data)
    
    
if __name__ == '__main__':
    data = scipy.io.loadmat('linear_regression.mat')

#     X = np.array([[1,2], 
#                   [3,4]])
#     print X.shape
#     print range(2)
#     X = np.dot(X, X)
#     print X
    
    #linear_reg(data)
    #ridge_reg(data)
    print range(-3, 3)
    
