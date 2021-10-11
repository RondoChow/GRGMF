'''
[1]Gu, Quanquan, and Jie Zhou. "Co-clustering on manifolds." Proceedings of the 15th ACM SIGKDD international conference
 on Knowledge discovery and data mining. ACM, 2009.
# @author: Zi-Chao Zhang
# @date:   2019.10.10
'''

import os
import numpy as np
from numpy import dot
from numpy.linalg import inv, pinv
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from time import time
import logging


class DRCC:
    def __init__(self, n=5, c=50, m=50, lam=0.1, miu=0.1, max_iter=50):
        self.n = n
        self.c = c
        self.m = m
        self.lam = lam
        self.miu = miu
        self.max_iter = max_iter

    def laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        # it is actually just a kind of symmetrization...@alfred 2018/7/28
        L = 0.5 * (np.diag(x + y) - (S + S.T))  # neighborhood regularization matrix
        return L

    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :], kind='mergesort')[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        np.random.seed(seed)
        t1 = time()
        num_drug, num_target = intMat.shape
        # ## sparsification
        self.drugMat = self.get_nearest_neighbors(drugMat, self.n)
        self.targetMat = self.get_nearest_neighbors(targetMat, self.n)

        X = intMat * W
        F = np.abs(np.random.randn(num_target, self.c))
        G = np.abs(np.random.randn(num_drug, self.m))
        Lg = self.laplacian_matrix(self.drugMat)
        Lf = self.laplacian_matrix(self.targetMat)
        S = self.update_S(G, F, X)
        for i in range(self.max_iter):
            S = self.update_S(G, F, X)
            G = self.update_G(Lg, X, G, S, F)
            F = self.update_F(Lf, X, G, S, F)
            loss = self.loss_fun(X, G, S, F, Lg, Lf)
            logging.debug('iter: %d, loss: %.4f' % (i, loss))
        self.predict = dot(dot(G, S), F.T)
        logging.info('time: %.2f' % (time() - t1))

    def update_S(self, G, F, X):
        Glinv = dot(pinv(dot(G.T, G)), G.T)
        Frinv = dot(F, pinv(dot(F.T, F)))
        return dot(dot(Glinv, X), Frinv)

    def postive(self, M):
        return (np.abs(M) + M) / 2

    def negative(self, M):
        return (np.abs(M) - M) / 2

    def update_G(self, Lg, X, G, S, F):
        P = dot(dot(X, F), S.T)
        Q = np.dot(np.dot(np.dot(S, F.T), F), S.T)

        P_p = self.postive(P)
        P_n = self.negative(P)
        Q_p = self.postive(Q)
        Q_n = self.negative(Q)
        Lg_p = self.postive(Lg)
        Lg_n = self.negative(Lg)
        return G * np.sqrt(
            (self.miu * dot(Lg_n, G) + P_p + dot(G, Q_n)) / (self.miu * dot(Lg_p, G) + P_n + dot(G, Q_p)))

    def update_F(self, Lf, X, G, S, F):
        A = dot(dot(X.T, G), S)
        B = np.dot(np.dot(np.dot(S.T, G.T), G), S)

        A_p = self.postive(A)
        A_n = self.negative(A)
        B_p = self.postive(B)
        B_n = self.negative(B)
        Lf_p = self.postive(Lf)
        Lf_n = self.negative(Lf)
        return F * np.sqrt(
            (self.lam * dot(Lf_n, F) + A_p + dot(F, B_n)) / (self.lam * dot(Lf_p, F) + A_n + dot(F, B_p)))

    def loss_fun(self, X, G, S, F, Lg, Lf):
        return (np.linalg.norm(X - dot(dot(G, S), F.T), 'fro') ** 2
                + self.lam * np.trace(dot(dot(F.T, Lf), F))
                + self.miu * np.trace(dot(dot(G.T, Lg), G)))

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predict[inx[:, 0], inx[:, 1]]

    def test(self, test_data):
        scores = self.predict[test_data[:, 0], test_data[:, 1]]
        return scores

    def evaluation(self, test_data, test_label):
        scores = self.predict[test_data[:, 0], test_data[:, 1]]
        if np.isnan(scores).any():
            logging.warning("Nan exists in prediction")
            scores[np.isnan(scores)] = 0.5
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def __str__(self):
        return "Model: DRCC, n:%s, c:%s, m:%s, lam:%s, miu:%s, max_iter:%s" % (
            self.n, self.c, self.m, self.lam, self.miu, self.max_iter)
