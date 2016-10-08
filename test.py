import numpy as np


def standalize_col(arr):
    for j in range(arr.shape[1]):
        print j
        cmean = np.mean(arr[:,j])
        print cmean
        cstd = np.std(arr[:,j])
        print cstd
        arr[:,j] = (arr[:,j]-cmean)/float(cstd)
    return arr

if __name__ == '__main__':
    X1 = np.array([[1.0,3,], 
                  [4,6,],
                  [7,1,]])
    X2 = np.array([[10], 
                  [11],
                  [12]])
#     #X = np.concatenate((X1, X2), axis=1)
#     p = np.random.permutation(len(X1))
#     print p
#     X1 = X1[p]
#     X2 = X2[p]
#     print X1
#     print X2
#     #print np.random.shuffle(X1, X2, random_state=0)

#     X1 = standalize_col(X1)
#     print X1
    print np.cov(X1.T)