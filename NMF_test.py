# coding: UTF-8
from sklearn.utils.extmath import randomized_svd, squared_norm
from sklearn.decomposition import NMF
from math import sqrt
import numpy as np
import pandas as pd
import csv


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

nmf = NMF(n_components=50)  # 设有2个隐主题
W = nmf.fit_transform(X)
H = nmf.components_
print X
print W
print H

recon = pd.DataFrame(np.dot(W,H),b0.index,b0.columns)
recon[recon<1] = 1
recon[recon>5] = 5
# recon.to_csv('nmfrecon.csv')
print recon
d = (t0 - recon) * (t0>0)
# d.to_csv('nmfd.csv')
d.fillna(0,inplace = True)
print squared_norm(d)