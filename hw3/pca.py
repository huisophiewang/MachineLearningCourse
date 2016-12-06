import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca(X, d):
    # change X back to NxD
    X = X.T
    N, D = X.shape
    X_mean = np.mean(X, axis=0)
    #print X_mean
    Z = np.zeros((D, N))
    for i in range(N):
        Z[:,i] = X[i] - X_mean
    #print Z
    Ux, Sx, Vx = np.linalg.svd(Z)
    U = Ux[:,:d]
    #print U
    mu = X_mean
    Y = np.zeros((N, d))
    for i in range(N):
        Y[i] = np.dot(U.T, (X[i]-X_mean))
    #print Y
    return U, mu, Y

def plot(Y):
    plt.plot(Y[:,0], Y[:,1], 'ro')
    plt.show()
    
def sklearn_pca(X, d):
    N, D = X.shape
    X_mean = np.mean(X, axis=0)
    #print X_mean
    Z = np.zeros((D, N))
    for i in range(N):
        Z[:,i] = X[i] - X_mean
        
    sklearn_pca = PCA(n_components=2, svd_solver='full')
    Y_sklearn = sklearn_pca.fit_transform(Z.T)
    #print Y_sklearn
    return Y_sklearn
    
if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = scipy.io.loadmat(os.path.join(cur_dir, 'data.mat'))
    # input must be X (DxN)
    U, mu, Y = pca(data['X_Question1'], d=2)
    plot(Y)


