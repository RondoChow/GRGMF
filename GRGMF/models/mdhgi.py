# -*- coding: utf-8 -*-
#Chen X, Yin J, Qu J, Huang L (2018)MDHGI: Matrix Decomposition and Heterogeneous Graph Inference for miRNA-disease association prediction. PLoS Comput Biol 14(8): e1006418
#added at 2018.11.19
import numpy.linalg as LA
import scipy.linalg as LG
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


class MDHGI:

    def __init__(self, alpha=0.1,a=0.4):

        self.alpha = float(alpha)
        self.mu = 10**-4
        self.rho = 1.1
        self.a = float(a)

        # read the weighted miRNA functional similarity
        # self.weightedRNA = np.loadtxt(r'..\data\datasets\hmdd2\weighted miRNA functional similarity.txt')#MDHGI在HMDD2上才会用到这个，其他情况默认权重都为1，也就是相似度都是已知的
        # read the weighted disease semantic similarity
        # self.wieghtedDis = np.loadtxt(r'..\data\datasets\hmdd2\weighted disease semantic similarity.txt')#MDHGI在HMDD2上才会用到这个，其他情况默认权重都为1，也就是相似度都是已知的


    def solve_l1l2(self,W,lambda1):
        nv = W.shape[1]#the number of columns in W
        F = W.copy()
        for p in range(nv):
            nw=LA.norm(W[:,p],"fro")
            if nw>lambda1:
                F[:,p]=(nw-lambda1)*W[:,p]/nw
            else:F[:,p]=np.zeros((W[:,p].shape[0],1))
        return F

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):      #copy from nrlmf.py ;should be suitable for this algorithm
        self.num_drugs, self.num_targets = intMat.shape
        self.weightedRNA = np.mat(np.ones((self.num_targets,self.num_targets)))
        self.wieghtedDis = np.mat(np.ones((self.num_drugs,self.num_drugs)))
        self.intMat = intMat * W       #this"self.intMat" discard the test_data.We train "self.intMat" not"intMat"
        self.X = np.mat(np.zeros((self.num_drugs, self.num_drugs)))

        self.ones = np.ones((self.num_drugs, self.num_targets))
        x, y = np.where(self.intMat > 0)
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        self.intMat = self.intMat.T     #!!!    495x383 fit the origin mdhgi.py

        P = self.MatrixDecomposition(self.intMat)
        SD, SM = self.GIP(P, drugMat, targetMat, self.wieghtedDis, self.weightedRNA)
        self.S = self.score(P, SD, SM)
        self.S = self.S.T               #输入算法之前转置了，这里转置回来


    def MatrixDecomposition(self,A):
        E = np.mat(np.zeros((self.num_targets, self.num_drugs)))
        J = np.mat(np.zeros((self.num_drugs, self.num_drugs)))
        Y1 = np.mat(np.zeros((self.num_targets, self.num_drugs)))
        Y2 = np.mat(np.zeros((self.num_drugs, self.num_drugs)))
        iter_num = 0
        # y_plot = np.zeros((50, 1), dtype=float)  # add for plot loss
        while True:
            [U, sigma1, V] = LG.svd(self.X + Y2 / self.mu, lapack_driver='gesvd')  # sigma1是一个向量（一维列表）
            G = [sigma1[k] for k in range(len(sigma1)) if sigma1[k] > 1 / self.mu]  # G取前k个奇异值列表，1/mu为阈值
            svp = len(G)
            if svp >= 1:
                sigma1 = sigma1[0:svp] - 1 / self.mu
            else:
                sigma1 = [0]
                svp = 1
            J = np.mat(U[:, 0:svp]) * np.mat(np.diag(sigma1)) * np.mat(V[0:svp, :])  # J是X+Y2/mu经过SVD恢复得到的 #A.T似乎是A的转置 #A.I是A的求逆
            ATA = np.dot(A.T, A)
            self.X = np.mat(ATA + np.eye(self.num_drugs)).I * (ATA - A.T * E + J + (A.T * Y1 - Y2) / self.mu)
            temp1 = A - A * self.X
            E = self.solve_l1l2(temp1 + Y1 / self.mu, self.alpha / self.mu)  # 对E的更新。看IALM算法流程中的3.c
            Y1 = Y1 + self.mu * (temp1 - E)
            Y2 = Y2 + self.mu * (self.X - J)
            self.mu = min(self.rho * self.mu, 10**10)  # 慢慢增加μ，max_mu为mu上限10**10

            # y_plot[iter_num] = LA.norm(temp1 - E, np.inf)  # add for plot loss
            iter_num = iter_num + 1
            if LA.norm(temp1 - E, np.inf) < 10**-6 and LA.norm(self.X - J, np.inf) < 10**-6 or iter_num>1500:
                # self.plot_loss(50, y_plot)  # add for plot loss
                break
        P = A * self.X
        return P

    def GIP(self,P,C,D,c,d):        #C-c-disease-drug   while   D-d-miRNA-target
        # calculate the Gaussian interaction profile kernel similarity KD and the integrated similarity SD for diseases
        gamad1 = 1
        sum1 = 0
        for nm in range(self.num_drugs):
            sum1 = sum1 + LA.norm(P[:, nm], "fro") ** 2
        gamaD1 = gamad1 * self.num_drugs / sum1
        KD = np.mat(np.zeros((self.num_drugs, self.num_drugs)))
        for ab in range(self.num_drugs):
            for ba in range(self.num_drugs):
                KD[ab, ba] = np.exp(-gamaD1 * LA.norm(P[:, ab] - P[:, ba], "fro") ** 2)
        SD = np.multiply((C + KD) * 0.5, c) + np.multiply(KD, 1 - c)  # 点乘

        # the normalization of SD
        SD1 = SD.copy()
        for nn1 in range(self.num_drugs):
            for nn2 in range(self.num_drugs):
                SD[nn1, nn2] = SD[nn1, nn2] / (np.sqrt(np.sum(SD1[nn1, :])) * np.sqrt(np.sum(SD1[nn2, :])))

        # calculate the Gaussian interaction profile kernel similarity KM and the integrated similarity SM for miRNAs
        gamad2 = 1
        sum2 = 0
        for mn in range(self.num_targets):
            sum2 = sum2 + LA.norm(P[mn, :], "fro") ** 2
        gamaD2 = gamad2 * self.num_targets / sum2
        KM = np.mat(np.zeros((self.num_targets, self.num_targets)))
        for cd in range(self.num_targets):
            for dc in range(self.num_targets):
                KM[cd, dc] = np.exp(-gamaD2 * LA.norm(P[cd, :] - P[dc, :], "fro") ** 2)
        SM = np.multiply((D + KM) * 0.5, d) + np.multiply(KM, 1 - d)

        # the normalization of SM
        SM1 = SM.copy()
        for mm1 in range(self.num_targets):
            for mm2 in range(self.num_targets):
                SM[mm1, mm2] = SM[mm1, mm2] / (np.sqrt(np.sum(SM1[mm1, :])) * np.sqrt(np.sum(SM1[mm2, :])))
        return SD,SM

    def score(self,P,SD,SM):
        S = np.mat(np.random.rand(self.num_targets, self.num_drugs))
        Si = self.a * SM * S * SD + (1-self.a) * P
        while LA.norm(Si - S, 1) > 10 ** -6:
            S = Si
            Si = self.a * SM * S * SD + (1-self.a) * P
        S = np.array(S)
        return S

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return np.array(self.S[inx[:, 0], inx[:, 1]])

    def evaluation(self, test_data, test_label):        #copy from nrlmf.py ;should be suitable for this algorithm
        scores = []
        for d, t in test_data:                          #test_data :383x495
            scores.append(self.S[d, t])
        prec, rec, thr = precision_recall_curve(test_label, np.array(scores))
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, np.array(scores))
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def plot_loss(self,max_iter,y_plot):
        plt.plot(np.arange(max_iter), y_plot, label='linear')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title("MDHGI")
        plt.show()

    def test(self, test_data):
        scores = self.S[test_data[:, 0], test_data[:, 1]]
        return scores

    def __str__(self):
        return "Model: MDHGI, alpha:%s, a:%s" %(self.alpha, self.a)
