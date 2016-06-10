# !/usr/bin/python2.7
# coding: UTF-8
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import sparse_nmf as smf

def sss_mf(X, n_components=2, max_iter=500,
          tol=1e-6, alpha=0.02, beta=0.01):

    M, N = np.shape(X)
    K = n_components

    W = np.random.rand(M,K)
    H = np.random.rand(K,N)
    print W
    print H

    for step in xrange(max_iter):

        eR = X - np.dot(W,H)
        W_temp = W

        for i in xrange(M):
            for k in xrange(K):
                sum = 0
                for j in xrange(N):
                    if X[i][j] > 0:
                        sum += eR[i][j] * H[k][j]
                W[i][k] = W[i][k] + alpha * (2 * sum - beta * W[i][k])

        if True:
            for j in xrange(N):
                for k in xrange(K):
                    sum = 0
                    for i in xrange(M):
                        if X[i][j] > 0:
                            sum += eR[i][j] * W_temp[i][k]

                    H[k][j] = H[k][j] + alpha * (2 * sum - beta * H[k][j])

        e = 0
        for i in xrange(M):
            for j in xrange(N):
                if X[i][j] > 0:
                    e = e + (X[i][j] - np.dot(W[i,:],H[:,j]))**2

        e = e + (beta/2) * ( (W**2).sum() + (H**2).sum() )

        if e < tol:
            break
    return W, H


RATE_MATRIX = np.array(
    [[5, 5, 3, 0, 5, 5],
     [5, 0, 4, 0, 4, 4],
     [0, 3, 0, 5, 4, 5],
     [5, 4, 3, 3, 0, 5]]
)

# nmf = NMF(n_components=2, init='random')  # 设有2个隐主题
# W = nmf.fit_transform(RATE_MATRIX)
# H = nmf.components_

mf = smf.Sparse_NMF(n_components=2, max_iter=20000, tol=0.4, alpha=0.01, beta=0.001, init='random')
W, H = mf.fit(RATE_MATRIX)
print mf.n_iter_
bob = np.array([5, 5, 0, 0, 0, 5])
bob.shape = 1,6
w = mf.transform(bob)

# W, H = sss_mf(RATE_MATRIX)

print W
print H
print np.dot(W,H)
print w
print np.dot(w,H)

# RATE = np.array(
#     [[5, 5, 3, 0, 5, 5],
#      [5, 0, 4, 0, 4, 4],
#      [0, 3, 0, 5, 4, 5],
#      [5, 4, 3, 3, 5, 5],
#      [5, 5, 0, 0, 0, 5]]
# )

# nmf1 = NMF(n_components=2)  # 设有2个隐主题
# user = nmf1.fit_transform(RATE)
# item = nmf1.components_
# print user
# print item
#
#
# print '用户的主题分布：'
# print user
# print '物品的主题分布：'
# print item
# print '重构矩阵：'
# print np.dot(user,item)
# print user_distribution
# bob = [5, 5, 0, 0, 0, 5]
# print np.array(bob)
# print 'Bob的主题分布：'
# bob_distribution = nmf.transform(bob)
# print bob_distribution
# print np.dot(user_distribution,item_distribution)
# print np.dot(bob_distribution,item_distribution)

users = ['1', '2', '3', '4']
zip_data = zip(users, W)

plt.title(u'the distribution of users')
plt.xlim((0, 3))
plt.ylim((0, 3.5))
for item in zip_data:
    user_name = item[0]
    data = item[1]
    plt.plot(data[0], data[1], "b*")
    plt.text(data[0], data[1], user_name, bbox=dict(facecolor='red', alpha=0.2),)

plt.plot(w[0][0], w[0][1], "b*")
plt.text(w[0][0], w[0][1], '5', bbox=dict(facecolor='green', alpha=0.2),)

plt.show()

# a = np.array([[1,2],
#      [2,2]])
# print a
# print a**2
# print (a**2).sum()
