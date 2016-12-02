import os
import math
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def random_initalize_clusters(data, k):
    clusters = []
    for i in range(data.shape[0]):
        c = random.randint(0, k-1)
        clusters.append(c)
    return clusters

# Euclidean distance of x1 and x2, k dimensions
def distance(x1, x2, k):
    dist = 0.0
    for i in range(k):
        dist += math.pow((x1[i]-x2[i]), 2)
    dist = math.sqrt(dist)
    return dist

def find_centers(data, clusters, k):
    n = data.shape[0]
    m = data.shape[1]
    
    centers = np.zeros([k, m-1])   
    nums = np.zeros(k)
    
    for i, c in enumerate(clusters):
        centers[c] += data[i][:-1]
        nums[c] += 1
        
    normalized_mutual_information(data, clusters, k)
    
    for i in range(k):
        centers[i] /= nums[i]

    return centers

def assign_new_clusters(data, centers, k):
    n = data.shape[0]
    m = data.shape[1]
    
    new_clusters = []
    
    for i in range(n):
        dists = []
        for j in range(k):
            dist = distance(data[i], centers[j], m-1)
            dists.append((dist, j))
        #print dists
        dists = sorted(dists, key=lambda x: x[0])
        new_clusters.append(dists[0][1])
        
    return new_clusters

def kmeans(X, k, max_iter):
    clusters = random_initalize_clusters(data, k)
    plot(data, clusters, k)
    
    for iter in range(max_iter):
        print iter
        
        centers = find_centers(data, clusters, k)
    
        new_clusters = assign_new_clusters(data, centers, k)
        print new_clusters
        plot(data, new_clusters, k)
        
           
        if new_clusters == clusters:
            break
        else:
            clusters = new_clusters

    plot(data, new_clusters, k)
    
def plot(X):
    plt.plot(X[:,0], X[:,1], 'ro')
    plt.show()
    
if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = scipy.io.loadmat(os.path.join(cur_dir, 'data.mat'))
    x = data['X_Question2_3']
    #print x.shape
    plot(x.T)