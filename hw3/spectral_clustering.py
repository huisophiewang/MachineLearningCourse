import os
import scipy.io
import numpy as np
from kmeans import kmeans, plot_clusters

def spectral(X, k, sigma):
    X = input.T
    N = len(X)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            W[i][j] = np.exp(-np.sum((X[i]-X[j])**2)/(sigma**2))       
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    w, v = np.linalg.eig(L)
    print w
    idx = np.argsort(w)[:k]
    H = v[:,idx]
    clusters = kmeans(H.T, k, 100)
    plot_clusters(X, clusters)
    
if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = scipy.io.loadmat(os.path.join(cur_dir, 'data.mat'))
    input = data['X_Question2_3']
    for sigma in [0.001, 0.01, 0.1, 1]:
        spectral(input, 4, sigma)