# coding: UTF-8
from numpy import *
import pandas as pd
import time
import datetime
import numpy as np
import csv
import sparse_nmf as smf
from sklearn.utils.extmath import randomized_svd, squared_norm


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
t.to_csv('tna.csv')
t0 = t.fillna(0)
t0.to_csv('t.csv')


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
b.to_csv('bna.csv')
b0 = b.fillna(0)
b0.to_csv('b.csv')



X = np.array(b0)
mf = smf.Sparse_NMF(n_components=50, max_iter=4000, tol=1000, alpha=0.002, beta=0.001, init='random')
W, H = mf.fit(X)
print mf.n_iter_
# bob = np.array([5, 5, 0, 0, 0, 5])
# bob.shape = 1,6
# w = mf.transform(bob)

# W, H = sss_mf(RATE_MATRIX)

# nmf = NMF(n_components=10, max_iter=500 )  # 设有2个隐主题
# W = nmf.fit_transform(np.array(f))
# H = nmf.components_

print W
print H
recon = pd.DataFrame(np.dot(W,H),b0.index,b0.columns)
recon[recon<1] = 1
recon[recon>5] = 5
# recon.to_csv('mf100recon.csv')
print recon
d = (t0 - recon) * (t0>0)
# d.to_csv('mf100d.csv')
d.fillna(0,inplace = True)
print squared_norm(d)

