from math import sqrt
import warnings
import numbers

import numpy as np
from sklearn.utils.extmath import randomized_svd, squared_norm
import scipy.sparse as sp


class Sparse_NMF():

    def __init__(self, n_components=2, init=None,
                 tol=1e-4, max_iter=200,
                 alpha=0.01, beta=0.01):
        self.n_components = n_components	#number of features
        self.init = init					#the init way of
        self.tol = tol						#the error tolerance
        self.max_iter = max_iter			#the maximum iteration
        self.alpha = alpha					#learning rate
        self.beta = beta					#L2 coefficient


    def fit(self, X):
        #given X, return W and H
        W, H, n_iter_ = matrix_factorization(
            X=X, n_components=self.n_components,
            init=self.init, update_H=True,
            tol=self.tol, max_iter=self.max_iter, alpha=self.alpha,
            beta=self.beta)

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter_

        return W, H

    def transform(self, X):
        #given X, calculate W with the H got during fit procedure
        W, _, n_iter_ = matrix_factorization(
            X=X, H=self.components_, n_components=self.n_components_,
            init=self.init, update_H=False,
            tol=self.tol, max_iter=self.max_iter, alpha=self.alpha,
            beta=self.beta)

        self.n_iter_ = n_iter_
        return W

def matrix_factorization(X, H=None, n_components=None,
                               init=None, update_H=True,
                               tol=1e-4, max_iter=200, alpha=0.01,
                               beta=0.02):

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    # check W and H, or initialize them
    if not update_H:
        W = np.zeros((n_samples, n_components))
    else:
        W, H = _initialize_nmf(X, n_components, init=init, eps=1e-6)

    print W
    print H

    n_iter = 0
    e_before = 0
    for step in xrange(max_iter):
        n_iter = step + 1
        print n_iter

        xs, ys = X.nonzero()    # the x index and y index of nonzero
        W_temp = W
        ER = X - np.dot(W,H)	# the error matrix

        for i in xrange(n_samples):
            for k in xrange(n_components):
                sum = 0
                for j in ys[xs==i]:
                    sum += ER[i][j] * H[k][j]

                t = W[i][k] + alpha * (sum - beta * W[i][k])
                if t < 0:
                    a = alpha
                    for l in xrange(10):
                        a /= 2
                        t = W[i][k] + a * (sum - beta * W[i][k])
                        if t >= 0:
                            break
                    if t < 0:
                        t = W[i][k]
                W[i][k] = t

        if update_H:
            for j in xrange(n_features):
                for k in xrange(n_components):
                    sum = 0
                    for i in xs[ys==j]:
                        sum += ER[i][j] * W_temp[i][k]

                    t = H[k][j] + alpha * (sum - beta * H[k][j])
                    if t < 0:
                        a = alpha
                        for l in xrange(10):
                            a /= 2
                            t = H[k][j] + a * (sum - beta * H[k][j])
                            if t >= 0:
                                break
                        if t < 0:
                            t = H[k][j]
                    H[k][j] = t

        E = (X - np.dot(W,H)) * (X>0)
        e = squared_norm(E) + beta * ( squared_norm(W) + squared_norm(H) )
        # if step > 0:
        #     if abs(e/e_before - 1) < tol:
        #         break
        # e_before = e
        print e
        if e < tol:
            break

    if n_iter == max_iter:
        print ("Maximum number of iteration %d reached. Increase it to"
                      " improve convergence." % max_iter)

    return W, H, n_iter

# def matrix_factorization(X, H=None, n_components=None,
#                                init='random', update_H=True,
#                                tol=1e-4, max_iter=200, alpha=0.01,
#                                beta=0.02):
#
#     n_samples, n_features = X.shape
#     if n_components is None:
#         n_components = n_features
#
#     # check W and H, or initialize them
#     if not update_H:
#         W = np.zeros((n_samples, n_components))
#     else:
#         W, H = _initialize_nmf(X, n_components, init=init, eps=1e-6)
#
#     n_iter = 0
#     e_before = 0
#     xs, ys = X.nonzero()
#
#     for step in xrange(max_iter):
#         n_iter = step + 1
#
#         V = np.dot(W,H)
#         W_temp = W
#
#         for i in xrange(n_samples):
#             for k in xrange(n_components):
#                 sum = 0
#                 whht = 0
#                 for j in ys[xs==i]:
#                     eij = X[i][j] - V[i][j]
#                     sum += eij * H[k][j]
#                     whht += V[i][j] * H[k][j]
#                 W[i][k] = W[i][k] + sum * W[i][k] / whht
#                 print W[i][k] / whht
#
#         if update_H:
#             for j in xrange(n_features):
#                 for k in xrange(n_components):
#                     sum = 0
#                     wtwh= 0
#                     for i in xs[ys==j]:
#                         eij = X[i][j] - V[i][j]
#                         sum += eij * W_temp[i][k]
#                         wtwh += W_temp[i][k] * V[i][j]
#                     H[k][j] = H[k][j] + sum * H[k][j] / wtwh
#                     print H[k][j] / wtwh
#
#         e = 0
#         for i in xrange(n_samples):
#             for j in ys[xs==i]:
#                 e = e + (X[i][j] - V[i][j])**2 / 2
#
#         # e = e + (beta/2) * ( (W*W).sum() + (H*H).sum() )
#         if step > 0:
#             if abs(e/e_before - 1) < tol:
#                 break
#         e_before = e
#
#     if n_iter == max_iter:
#         print ("Maximum number of iteration %d reached. Increase it to"
#                       " improve convergence." % max_iter)
#
#     return W, H, n_iter

# def matrix_factorization(X, H=None, n_components=None,
#                                init='random', update_H=True,
#                                tol=1e-4, max_iter=200, alpha=0.01,
#                                beta=0.02):
#
#     n_samples, n_features = X.shape
#     if n_components is None:
#         n_components = n_features
#
#     # check W and H, or initialize them
#     if not update_H:
#         W = np.zeros((n_samples, n_components))
#     else:
#         W, H = _initialize_nmf(X, n_components, init=init, eps=1e-6)
#
#     n_iter = 0
#     e_before = 0
#     xs, ys = X.nonzero()
#
#     for step in xrange(max_iter):
#         n_iter = step + 1
#
#         V = np.dot(W,H)
#         W_temp = W
#
#         for i in xrange(n_samples):
#             for k in xrange(n_components):
#                 sum = 0
#                 whht = 0
#                 for j in ys[xs==i]:
#                     sum += X[i][j] * H[k][j]
#                     whht += V[i][j] * H[k][j]
#                 W[i][k] = sum * W[i][k] / whht
#
#         if update_H:
#             for j in xrange(n_features):
#                 for k in xrange(n_components):
#                     sum = 0
#                     wtwh= 0
#                     for i in xs[ys==j]:
#                         sum += W_temp[i][k] * X[i][j]
#                         wtwh += W_temp[i][k] * V[i][j]
#                     H[k][j] = sum * H[k][j] / wtwh
#
#         e = 0
#         for i in xrange(n_samples):
#             for j in ys[xs==i]:
#                 e = e + (X[i][j] - V[i][j])**2 / 2
#
#         # e = e + (beta/2) * ( (W*W).sum() + (H*H).sum() )
#         if step > 0:
#             if abs(e/e_before - 1) < tol:
#                 break
#         e_before = e
#
#     if n_iter == max_iter:
#         print ("Maximum number of iteration %d reached. Increase it to"
#                       " improve convergence." % max_iter)
#
#     return W, H, n_iter

# def matrix_factorization(X, H=None, n_components=None,
#                                init='random', update_H=True,
#                                tol=1e-4, max_iter=200, alpha=0.01,
#                                beta=0.02):
#
#     n_samples, n_features = X.shape
#     if n_components is None:
#         n_components = n_features
#
#     # check W and H, or initialize them
#     if not update_H:
#         W = np.zeros((n_samples, n_components))
#     else:
#         W, H = _initialize_nmf(X, n_components, init=init, eps=1e-6)
#
#     n_iter = 0
#     e_before = 0
#     xs, ys = X.nonzero()
#
#     for step in xrange(max_iter):
#         n_iter = step + 1
#
#         V = np.dot(W,H)
#         W_temp = W
#
#         for i in xrange(n_samples):
#             for k in xrange(n_components):
#                 s1 = 0
#                 s2 = 0
#                 for j in xrange(n_features):
#                     s1 += X[i][j] * H[k][j] / V[i][j]
#                     s2 += H[k][j]
#                 W[i][k] = s1 * W[i][k] / s2
#
#         if update_H:
#             for j in xrange(n_features):
#                 for k in xrange(n_components):
#                     s1 = 0
#                     s2 = 0
#                     for i in xrange(n_samples):
#                         s1 += W[i][k] * X[i][j] / V[i][j]
#                         s2 += W_temp[i][k]
#                     H[k][j] = s1 * H[k][j] / s2
#
#         e = 0
#         for i in xrange(n_samples):
#             for j in ys[xs==i]:
#                 e = e + (X[i][j] - V[i][j])**2 / 2
#
#         # e = e + (beta/2) * ( (W*W).sum() + (H*H).sum() )
#         if step > 0:
#             if abs(e/e_before - 1) < tol:
#                 break
#         e_before = e
#
#     if n_iter == max_iter:
#         print ("Maximum number of iteration %d reached. Increase it to"
#                       " improve convergence." % max_iter)
#
#     return W, H, n_iter


def _initialize_nmf(X, n_components, init=None, eps=1e-6):
    n_samples, n_features = X.shape

    if init is None:
        if n_components < n_features and n_components < n_samples:
            init = 'nndsvd'
        else:
            init = 'random'

    # Random initialization
    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)
        W = avg * np.random.randn(n_samples, n_components)
        H = avg * np.random.randn(n_components, n_features)
        # we do not write np.abs(H, out=H) to stay compatible with
        # numpy 1.5 and earlier where the 'out' keyword is not
        # supported as a kwarg on ufuncs
        np.abs(H, H)
        np.abs(W, W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        avg = X.mean()
        W[W == 0] = abs(avg * np.random.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * np.random.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    return W, H

def norm(x):
    return sqrt(squared_norm(x))