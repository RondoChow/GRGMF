'''
[1] Xiao, Q.,Luo J.W. et al, (2017),A Graph Regularized Non-negative Matrix Factorization,
# @author: Zi-Chao Zhang
# @date:   2019.10.12

note: jave SE, matlab, and the package "matlab" for python are required
'''

import os
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from time import time
import matlab.engine  # cd "matlabroot/extern/engines/python"  python setup.py install
import logging
from functions import BASE_DIR


class GRNMF:
    def __init__(self, k=50, lambda_l=0.5, lambda_m=0.5, lambda_d=0.5, r=0.5, K=5, max_iter=30):
        self.K = K
        self.r = r
        self.p = 5
        self.k = k  # latent vector length
        self.dataset_n = 0
        self.max_iter = max_iter
        self.lambda_l = lambda_l
        self.lambda_m = lambda_m
        self.lambda_d = lambda_d
        self.mlab = None

    def __del__(self):
        if self.mlab:
            self.mlab.quit()

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        W = W
        self.num_drugs, self.num_targets = intMat.shape
        x, y = np.where(W > 0)
        self.train_drugs = set(x.tolist())
        self.train_targets = set(y.tolist())
        WR = W * intMat
        self.drugMat = (drugMat + drugMat.T) / 2
        self.targetMat = (targetMat + targetMat.T) / 2
        if self.mlab:
            pass
        else:
            # self.mlab = matlab.engine.start_matlab("-desktop")
            self.mlab = matlab.engine.start_matlab()
        t = time()
        self.mlab.addpath(os.path.realpath(os.path.join(BASE_DIR, 'GRNMF')))
        self.predictR = self.mlab.grnmf_fixmodel(
            {'realpath': os.path.realpath(os.path.join(BASE_DIR, 'GRNMF')), 'drugMat': matlab.double(self.drugMat.tolist()),
             'targetMat': matlab.double(self.targetMat.tolist()), 'WR': matlab.double(WR.tolist()), 'K': self.K,
             'r': self.r, 'p': self.p, 'k': self.k, 'max_iter': self.max_iter, 'lambda': self.lambda_l,
             'lambda_m': self.lambda_m, 'lambda_d': self.lambda_d})
        self.predictR = np.array(self.predictR)
        logging.info("dataset: %d, t: %.2f" % (self.dataset_n, time() - t))
        self.dataset_n += 1

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predictR[inx[:, 0], inx[:, 1]]

    def test(self, test_data):
        scores = self.predictR[test_data[:, 0], test_data[:, 1]]
        return scores

    def evaluation(self, test_data, test_label):
        scores = self.predictR[test_data[:, 0], test_data[:, 1]]
        scores[np.isnan(scores)] = 0.5
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        logging.info("dataset: %d, auc: %.3f, aupr: %.3f" % (self.dataset_n, auc_val, aupr_val))
        return aupr_val, auc_val

    def __str__(self):
        return "Model: GRNMF, k:%s, lambda:%s, lambda_m:%s, lambda_d:%s, K: %s, max_iter:%s" % (
            self.k, self.lambda_l, self.lambda_m, self.lambda_d, self.K, self.max_iter)
