# -*- coding: utf-8 -*-
'''
[CMF] X. Zheng, H. Ding, H. Mamitsuka, and S. Zhu, "Collaborative matrix factorization with multiple similarities for predicting drug-target interaction", KDD, 2013.
[CMFMDA] Zhen Shen, You-Hua Zhang, Kyungsook Han, Asoke K. Nandi"miRNA-Disease Association Prediction with Collaborative Matrix Factorization" 2017
对比CMF，CMFMDA 中其U、V 的初始值是由Y 经过 WKNKN 函数的处理后再 svd 分解后得到的。矩阵分解代码参照论文中的损失函数、迭代式，代码与CMF 中的稍有不同
'''
import numpy as np
import scipy.linalg as LG
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from functions import WKNKN

class CMFMDA:

    def __init__(self,K=10, ita=0.5, k_dimen=10, lambda_l=0.01, lambda_d=0.01, lambda_t=0.01, max_iter=100):
        self.K = int(K)                     # WKNKN 取最相似的前 K 个邻居
        self.ita = float(ita)
        self.k_dimen = int(k_dimen)         # 矩阵分解降的维度
        self.lambda_l = float(lambda_l)
        self.lambda_d = float(lambda_d)
        self.lambda_t = float(lambda_t)
        self.max_iter = int(max_iter)

    def fix_model(self, W, intMat, drugMat, targetMat, seed):
        self.num_drugs, self.num_targets = intMat.shape
        self.drugMat, self.targetMat = drugMat, targetMat
        x, y = np.where(W > 0)
        self.train_drugs = set(x.tolist())
        self.train_targets = set(y.tolist())

        # if seed is None:
        #     self.U = np.sqrt(1/float(self.K))*np.random.normal(size=(self.num_drugs, self.K))
        #     self.V = np.sqrt(1/float(self.K))*np.random.normal(size=(self.num_targets, self.K))
        # else:
        #     prng = np.random.RandomState(seed)
        #     self.U = np.sqrt(1/float(self.K))*prng.normal(size=(self.num_drugs, self.K))
        #     self.V = np.sqrt(1/float(self.K))*prng.normal(size=(self.num_targets, self.K))

        self.ones = np.identity(self.k_dimen)
        WR = W*intMat
        WR_new = WKNKN(WR, drugMat, targetMat, self.K, self.ita)
        y_plot = np.zeros((self.max_iter, 1), dtype=float)  # add for plotting loss
        [U, sigma1, V] = LG.svd(WR_new)

        self.U = U[:, 0:self.k_dimen].dot(np.diag(np.sqrt(sigma1[0:self.k_dimen])))      #d x k
        self.V = V[0:self.k_dimen, :].T.dot(np.diag(np.sqrt(sigma1[0:self.k_dimen])))    #t x k

        last_loss = self.compute_loss(W, intMat, drugMat, targetMat)
        for t in range(self.max_iter):
            self.U = self.als_update(self.U, self.V, W, WR_new, drugMat, self.lambda_l, self.lambda_d)
            self.V = self.als_update(self.V, self.U, W.T, WR_new.T, targetMat, self.lambda_l, self.lambda_t)
            y_plot[t] = last_loss  # add for plotting loss
            curr_loss = self.compute_loss(W, intMat, drugMat, targetMat)
            delta_loss = (curr_loss-last_loss)
            # print "Epoach:%s, Curr_loss:%s, Delta_loss:%s" % (t+1, curr_loss, delta_loss)
            if abs(delta_loss) < 1e-6:
                break
            last_loss = curr_loss
        # self.plot_loss(self.max_iter, y_plot)  # add for plotting loss


    def als_update(self, A, B, W, Y, S, lambda_l, lambda_d):    #modified
        C = Y.dot(B)+lambda_d * S.dot(A)
        D = B.T.dot(B)+lambda_l * self.ones + lambda_d * A.T.dot(A)
        U0= C.dot(np.linalg.inv(D))

        return U0

    def compute_loss(self, W, intMat, drugMat, targetMat):      #modified #todo 有问题：明明
        loss = np.linalg.norm(W * intMat - np.dot(self.U, self.V.T), "fro")**(2)
        loss += self.lambda_l*(np.linalg.norm(self.U, "fro")**(2)+np.linalg.norm(self.V, "fro")**(2))
        loss += self.lambda_d*np.linalg.norm(drugMat-self.U.dot(self.U.T), "fro")**(2)+self.lambda_t*np.linalg.norm(targetMat-self.V.dot(self.V.T), "fro")**(2)
        return 0.5*loss

    def evaluation(self, test_data, test_label):
        ii, jj = test_data[:, 0], test_data[:, 1]
        scores = np.sum(self.U[ii, :]*self.V[jj, :], axis=1)
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return np.sum(self.U[inx[:, 0], :]*self.V[inx[:, 1], :], axis=1)

    def plot_loss(self,max_iter,y_plot):
        plt.plot(np.arange(max_iter), y_plot, label='linear')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title("NMF")
        # plt.ion()
        # plt.pause(0.1)
        plt.show()

    def test(self, test_data):
        ii, jj = test_data[:, 0], test_data[:, 1]
        scores = np.sum(self.U[ii, :]*self.V[jj, :], axis=1)
        return scores

    def __str__(self):
        return "Model: CMFMDA, K:%s, ita:%s, k_dimen:%s, lambda_l:%s, lambda_d:%s, lambda_t:%s, max_iter:%s" % (self.K, self.ita, self.k_dimen, self.lambda_l, self.lambda_d, self.lambda_t, self.max_iter)
