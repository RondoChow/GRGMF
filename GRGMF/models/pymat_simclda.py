
'''
[1] Chengqian Lu, "Prediction of lncRNA-disease associations based on inductive matrix completion",
# @author: lufan
# @date:   2019.1.16
'''

import os
import numpy as np
from pymatbridge import Matlab
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class SIMCLDA:

    def __init__(self,lambda_r= 0.1, alpha_l=0.5, alpha_d=0.5, max_iter=100):
        self.lambda_r = lambda_r
        self.alpha_l = alpha_l
        self.alpha_d = alpha_d
        self.max_iter = max_iter

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        #
        #
        intMat = intMat.T
        W = W.T
        #
        # targetMat : nd x nd
        # drugMat   : nl x nl

        self.num_targets, self.num_drugs = intMat.shape         # shape=(nd,nl)  = (dis,lnc)
        self.drugMat, self.targetMat = drugMat, targetMat
        x, y = np.where(W > 0)
        self.train_targets = set(x.tolist())
        self.train_drugs = set(y.tolist())
        # intMat above have been transposed

        WR = W * intMat
        drugMat = (drugMat + drugMat.T) / 2
        targetMat = (targetMat + targetMat.T) / 2
        mlab = Matlab()
        mlab.start()
        # print os.getcwd()
        # self.predictR = mlab.run_func(os.sep.join([os.getcwd(), "kbmf2k", "kbmf.m"]), {'Kx': drugMat, 'Kz': targetMat, 'Y': R, 'R': self.num_factors})['result']
        self.predictR = mlab.run_func(os.path.realpath(os.sep.join(['../simclda', "simclda.m"])), {'drugMat': drugMat, 'targetMat': targetMat, 'WR': WR, 'lr': self.lambda_r, 'al': self.alpha_l, 'ad': self.alpha_d, 'max_iter': self.max_iter})['result']
        # don't know how I fix the error "utf-8"
        # print os.path.realpath(os.sep.join(['../kbmf2k', "kbmf.m"]))
        mlab.stop()

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predictR[inx[:, 1], inx[:, 0]]

    def test(self, test_data):
        scores = self.predictR[test_data[:, 1], test_data[:, 0]]
        return scores

    def evaluation(self, test_data, test_label):
        scores = self.predictR[test_data[:, 1], test_data[:, 0]]
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def __str__(self):
        return "Model: SIMCLDA, lambda_r:%s, alpha_l:%s, alpha_d:%s, max_iter:%s" % (self.lambda_r, self.alpha_l, self.alpha_d, self.max_iter)
