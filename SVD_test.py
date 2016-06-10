# coding: UTF-8
from sklearn.utils.extmath import randomized_svd, squared_norm
from math import sqrt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy import sparse
import csv

# def svd_mf(X, n_components, init=None, eps=1e-6):
#     n_samples, n_features = X.shape
#
#     if init is None:
#         if n_components < n_features and n_components < n_samples:
#             init = 'nndsvd'
#         else:
#             init = 'random'
#
#     # Random initialization
#     if init == 'random':
#         avg = np.sqrt(X.mean() / n_components)
#         W = avg * np.random.randn(n_samples, n_components)
#         H = avg * np.random.randn(n_components, n_features)
#         # we do not write np.abs(H, out=H) to stay compatible with
#         # numpy 1.5 and earlier where the 'out' keyword is not
#         # supported as a kwarg on ufuncs
#         np.abs(H, H)
#         np.abs(W, W)
#         return W, H
#
#     # NNDSVD initialization
#     U, S, V = randomized_svd(X, n_components)
#     W, H = np.zeros(U.shape), np.zeros(V.shape)
#
#     # The leading singular triplet is non-negative
#     # so it can be used as is for initialization.
#     W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
#     H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])
#
#     for j in range(1, n_components):
#         x, y = U[:, j], V[j, :]
#
#         # extract positive and negative parts of column vectors
#         x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
#         x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
#
#         # and their norms
#         x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
#         x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)
#
#         m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
#
#         # choose update
#         if m_p > m_n:
#             u = x_p / x_p_nrm
#             v = y_p / y_p_nrm
#             sigma = m_p
#         else:
#             u = x_n / x_n_nrm
#             v = y_n / y_n_nrm
#             sigma = m_n
#
#         lbd = np.sqrt(S[j] * sigma)
#         W[:, j] = lbd * u
#         H[j, :] = lbd * v
#
#     W[W < eps] = 0
#     H[H < eps] = 0
#
#     if init == "nndsvd":
#         pass
#     elif init == "nndsvda":
#         avg = X.mean()
#         W[W == 0] = avg
#         H[H == 0] = avg
#     elif init == "nndsvdar":
#         avg = X.mean()
#         W[W == 0] = abs(avg * np.random.randn(len(W[W == 0])) / 100)
#         H[H == 0] = abs(avg * np.random.randn(len(H[H == 0])) / 100)
#     else:
#         raise ValueError(
#             'Invalid init parameter: got %r instead of one of %r' %
#             (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))
#
#     return W, H

def norm(x):
    return sqrt(squared_norm(x))

def vector_to_diagonal(vector):
    """
    将向量放在对角矩阵的对角线上
    :param vector:
    :return:
    """
    if (isinstance(vector, np.ndarray) and vector.ndim == 1) or \
            isinstance(vector, list):
        length = len(vector)
        diag_matrix = np.zeros((length, length))
        np.fill_diagonal(diag_matrix, vector)
        return diag_matrix
    return None


file_path = "C:/Users/Wu/Desktop/ml-100k/u1.test"
f = open(file_path,'rb')
rows = csv.reader(f)
dic = {}
for row in rows:
    e = row[0].split('\t')
    if not dic.has_key(int(e[0])):
        dic[int(e[0])] = {}
    dic[int(e[0])][int(e[1])] =int(e[2])
temp = []
for key in dic:
    s = pd.Series(dic[key],name=key)
    temp.append(s)
t = pd.DataFrame(temp)
print t
t0 = t.fillna(0)


file_path = "C:/Users/Wu/Desktop/ml-100k/u1.base"
# file_path = "C:/Users/Wu/Desktop/ml-1m/ratings.dat"

f = open(file_path,'rb')
rows = csv.reader(f)

dic = {}
for row in rows:
    e = row[0].split('\t')
    if not dic.has_key(int(e[0])):
        dic[int(e[0])] = {}
    dic[int(e[0])][int(e[1])] =int(e[2])

temp = []
for key in dic:
    if dic[key].__len__() > 100:
        continue

    s = pd.Series(dic[key],name=key)
    temp.append(s)

# print s
b = pd.DataFrame(temp)
print b
b0 = b.fillna(0)



X = np.array(b0)
# print X.mean()
# print (X[X>0]).mean()
# X[X==0] = (X[X>0]).mean()
# print X

# U, S, V = randomized_svd(X, 20)
U, S, V = svds(sparse.csr_matrix(X),  k=50, maxiter=2000)

S = vector_to_diagonal(S)
print X
print U
print S
print V

recon= pd.DataFrame(np.dot(U,np.dot(S,V)),b0.index,b0.columns)
recon[recon<1] = 1
recon[recon>5] = 5
# recon.to_csv('svdrecon.csv')
print recon
d = (t0 - recon) * (t0>0)
# d.to_csv('svdd.csv')
d.fillna(0,inplace = True)
print squared_norm(d)