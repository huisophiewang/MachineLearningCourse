import os
import math
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def random_initalize_clusters(X, k):
    N, D = X.shape
    clusters = []
    for i in range(N):
        c = random.randint(0, k-1)
        clusters.append(c)
    return clusters

def distance(x1, x2, D):
    dist = 0.0
    for i in range(D):
        dist += (x1[i]-x2[i])**2
    dist = math.sqrt(dist)
    return dist

def find_centers(X, clusters, k):
    N, D = X.shape
    centers = np.zeros([k, D])   
    nums = np.zeros(k)    
    for i, c in enumerate(clusters):
        centers[c] += X[i]
        nums[c] += 1
    for i in range(k):
        centers[i] /= nums[i]
    return centers

def assign_new_clusters(X, centers, k):
    N, D = X.shape
    new_clusters = []
    for i in range(N):
        dists = []
        for j in range(k):
            dist = distance(X[i], centers[j], D)
            dists.append((dist, j))
        dists = sorted(dists, key=lambda x: x[0])
        new_clusters.append(dists[0][1])
    return new_clusters

def kmeans(input, k, max_iter):
    # change X back to NxD
    X = input.T
    clusters = random_initalize_clusters(X, k)
    plot_clusters(X, clusters)  
    for iter in range(max_iter):
        print iter
        centers = find_centers(X, clusters, k)
        new_clusters = assign_new_clusters(X, centers, k)
        #print new_clusters
        #plot_clusters(X, new_clusters)
        if new_clusters == clusters:
            print "Converged!"
            break
        else:
            clusters = new_clusters 
    print new_clusters       
    #plot_clusters(X, new_clusters)
    return new_clusters
       
def plot_clusters(X, clusters):
    N = len(X)
    colors = []
    for i in range(N):
        if clusters[i] == 0:
            colors.append('r')
        elif clusters[i] == 1:
            colors.append('g')
        elif clusters[i] == 2:
            colors.append('b')
        elif clusters[i] == 3:
            colors.append('y')
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.show()
    
   
if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = scipy.io.loadmat(os.path.join(cur_dir, 'data.mat'))
    input = data['X_Question2_3']
    # input must be X (DxN)
    kmeans(input, k=4, max_iter=100)
    