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
    

def linear_reg():
    data = scipy.io.loadmat('linear_regression.mat')
    #print data
    x_train, y_train = data['X_trn'], data['Y_trn']
    x_test, y_test = data['X_tst'], data['Y_tst']
    
    #degree = [2, 5, 10, 20]
    
if __name__ == '__main__':
    #linear_reg()
    X = np.array([[1,2], 
                  [3,4]])
    print X.shape
    print range(2)
    poly_basis_trans(X, 2)
