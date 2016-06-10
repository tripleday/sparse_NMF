# coding: UTF-8
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

W = [[ 1.38546147,  2.44913029],
 [ 2.6676311,   0.49082458],
 [ 2.64389252,  0.93286929],
 [ 1.61904192,  1.97489948]]
H = [[ 1.67764528,  0.51198896,  1.42412423,  1.86880929,  1.15084589,  1.33280225],
 [ 1.11501777,  1.70131603,  0.39102091,  0.01667151,  1.35925223,  1.35603383]]

users = ['U1', 'U2', 'U3', 'U4']
zip_data = zip(users, W)

plt.title(u'the distribution of users')
plt.xlim((0, 3.5))
plt.ylim((0, 3.5))
for item in zip_data:
    user_name = item[0]
    data = item[1]
    plt.plot(data[0], data[1], "b*")
    plt.text(data[0], data[1], user_name, bbox=dict(facecolor='red', alpha=0.2),)

w = [[ 1.29166668,  2.50157964]]
plt.plot(w[0][0], w[0][1], "b.")
plt.text(w[0][0], w[0][1], 'U5', bbox=dict(facecolor='green', alpha=0.2),)

plt.show()