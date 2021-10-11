# -*- coding: utf-8 -*-
# @author: Zichao Zhang
# @date:  July 2018

# TODO@Alfred: Tuning Adam Optimizer
# TODO@Alfred: Visualize loss
# TODO: pytorch

import sys

sys.path.append('../')
import numpy as np
from numpy import dot
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc as AUC
from time import time, asctime
import torch
from torch import mm, mul
import logging
from functions import WKNKN

datatype = torch.float32  # note: the default data type in numpy is float64. Here we use float32 to accelerate
# the computation, which make the result of torch-based model sligtly different from numpy-based one.
npeps = np.finfo(float).eps
add_eps = 0.01


# eps = torch.tensor(data=npeps, dtype=datatype, device=DEVICE)


class ADPGMF_gpu:
    def __init__(self, K=5, max_iter=100, lr=0.01, lamb=0.1, mf_dim=50, beta=0.1, r1=0.5, r2=0.5, c=5, pre_end=True,
                 cvs=None, verpose=10, resample=0, ita=0):
        '''Initialize the instance

        Args:
            K: Num of neighbors
            max_iter: Maximum num of iteration
            lr: learning rate
            lamb: trade-off parameter for norm-2 regularization for matrix U and V
            mf_dim: dimension of the subspace expanded by self-representing vectors(i.e. the dimension for MF)
            beta: trade-off parameter for norm-2 regularization for matrix U and V
            r1: trade-off parameter of graph regularization for nodes in A
            r2: trade-off parameter of graph regularization for nodes in B
            c: constant of the important level for positive sample
            cvs: cross validation setting (1, 2 or 3)
            verpose: verpose level(for standard output) TODO: remove in the future
            resample: weather to resample the positive samples or not
        '''
        if torch.cuda.is_available():
            self.DEVICE = 'cuda'
            logging.warning('CUDA is available, DEVICE=cuda')
        else:
            self.DEVICE = 'cpu'
            logging.warning('CUDA is not available, DEVICE=cpu')
        self.K = K  # number of neighbors
        self.n = -1  # data tag
        self.mf_dim = mf_dim
        self.num_factors = mf_dim
        self.max_iter = max_iter
        self.lr = lr
        self.lamb = lamb
        self.beta = beta
        self.r1 = r1
        self.r2 = r2
        self.c = c
        self.loss = [[np.inf] for i in range(50)]
        self.cvs = cvs
        self.verpose = verpose
        self.resample = resample
        self.WK = K  # let "K" in WKNKN be the same as K
        self.ita = ita
        if cvs:
            if cvs == 1:
                self.imp1 = 3.
                self.imp2 = 2.
            elif cvs == 2:
                self.imp1 = 5.
                self.imp2 = 2.
            elif cvs == 3:
                self.imp1 = 3.
                self.imp2 = 4.
        else:
            self.imp1 = 3.
            self.imp2 = 2.

    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :], kind='mergesort')[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        '''Train the MF model

            Y=sigmoid(AUV^TB)
            (i.e. Y = sigmoid(Z^A U V^T Z^B))

        Args:
            W: Mask for training set
            intMat: complete interaction matrix
            drugMat: similarity matrix for nodes in A
            targetMat: similarity matrix for nodes in B
            seed: random seed to determine a random state

        Returns:
            None
        '''

        def loss_function(Y, W):
            '''
            Return the current value of loss function
            Args:
                Y: interaction matrix
                W: mask for training set

            Returns:
                current value of loss function
            '''
            temp = mm(self.A, mm(self.U, mm(self.V.t(), self.B)))  # avoiding overflow
            logexp = torch.zeros(temp.shape, dtype=datatype, device=DEVICE)
            logexp[temp > 50] = temp[temp > 50] * 1.  # np.log(np.e)
            logexp[temp <= 50] = torch.log(torch.exp(temp[temp <= 50]) + 1)

            loss = (mul(mul((1 + c * Y - Y), logexp)
                        - mul(c * Y, mm(self.A, mm(self.U, mm(self.V.t(), self.B)))), W).sum()
                    + lamb * (torch.norm(self.U, 2) ** 2 + torch.norm(self.V, 2) ** 2)
                    + beta * (torch.norm(self.A, 2) ** 2 + torch.norm(self.B, 2) ** 2)
                    + r1 * torch.trace(
                        mm(self.U.t(), mm(self.A.t(), mm(self.lap_mat(self.drugMat), mm(self.A, self.U)))))
                    + r2 * torch.trace(
                        mm(self.V.t(), mm(self.B, mm(self.lap_mat(self.targetMat), mm(self.B.t(), self.V))))))
            return loss

        # data split
        Y = intMat * W
        self.num_drugs, self.num_targets = Y.shape
        if (self.DEVICE == 'cuda') & (np.sqrt(self.num_drugs * self.num_targets) < 300):
            self.DEVICE = 'cpu'
            logging.warning('Matrix dimensions are small, DEVICE=CPU')
        DEVICE = self.DEVICE

        tt = time()
        self.n += 1
        self.drugMat = drugMat.copy()
        self.targetMat = targetMat.copy()

        # emphasize the diag of similarity matrix
        self.drugMat += self.imp1 * np.diag(np.diag(np.ones(drugMat.shape)))
        self.targetMat += self.imp2 * np.diag(np.diag(np.ones(targetMat.shape)))

        # ## sparsification
        self.drugMat = self.get_nearest_neighbors(self.drugMat, self.K + 1)
        self.targetMat = self.get_nearest_neighbors(self.targetMat, self.K + 1)

        # symmetrization
        self.drugMat = (self.drugMat + self.drugMat.T) / 2.
        self.targetMat = (self.targetMat + self.targetMat.T) / 2.

        # normalization
        self.A = dot(np.diag(1. / np.sum(self.drugMat + add_eps, axis=1).flatten()), self.drugMat + add_eps)
        self.B = dot(self.targetMat + add_eps, np.diag(1. / np.sum(self.targetMat + add_eps, axis=0).flatten()))

        # 2 initialization for U and V
        prng = np.random.RandomState(seed)
        if seed != None:
            self.U = np.sqrt(1. / float(self.num_factors)) * prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1. / float(self.num_factors)) * prng.normal(size=(self.num_targets, self.num_factors))
        else:
            self.U = np.sqrt(1. / float(self.num_factors)) * np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1. / float(self.num_factors)) * np.random.normal(size=(self.num_targets, self.num_factors))
        u, s, v = np.linalg.svd(WKNKN(Y, drugMat, targetMat, self.WK, self.ita))
        self.U[:, : min(self.num_factors, min(Y.shape))] = u[:, : min(self.num_factors, min(Y.shape))]
        self.V[:, : min(self.num_factors, min(Y.shape))] = v[:, :min(self.num_factors, min(Y.shape))]
        del u, s, v

        # load variables to GPU memory
        self.U = torch.tensor(self.U, dtype=datatype, device=DEVICE)
        self.V = torch.tensor(self.V, dtype=datatype, device=DEVICE)
        self.A = torch.tensor(self.A, dtype=datatype, device=DEVICE)
        self.B = torch.tensor(self.B, dtype=datatype, device=DEVICE)
        self.drugMat = torch.tensor(self.drugMat, dtype=datatype, device=DEVICE)
        self.targetMat = torch.tensor(self.targetMat, dtype=datatype, device=DEVICE)
        Y = torch.tensor(Y, dtype=datatype, device=DEVICE)
        W_all = torch.tensor(W, dtype=datatype, device=DEVICE)
        # W_all = torch.tensor(np.ones(W.shape), dtype=datatype, device=DEVICE)
        lamb = torch.tensor(data=self.lamb, dtype=datatype, device=DEVICE)
        beta = torch.tensor(data=self.beta, dtype=datatype, device=DEVICE)
        r1 = torch.tensor(data=self.r1, dtype=datatype, device=DEVICE)
        r2 = torch.tensor(data=self.r2, dtype=datatype, device=DEVICE)
        c = torch.tensor(data=self.c, dtype=datatype, device=DEVICE)
        max_iter = torch.tensor(data=self.max_iter, dtype=torch.int, device=DEVICE)
        eps = torch.tensor(data=npeps, dtype=datatype, device=DEVICE)

        # Using optimizer:
        lr = self.lr
        opter = self.adam_opt
        patient = 3
        numiter = torch.tensor(data=0, dtype=torch.int, device=DEVICE)
        numiter.copy_(max_iter)
        minloss = np.inf

        # store the initial value of A B U V for later use
        init_U = torch.zeros(self.U.shape, dtype=datatype, device=DEVICE)
        init_U.copy_(self.U)
        init_V = torch.zeros(self.V.shape, dtype=datatype, device=DEVICE)
        init_V.copy_(self.V)
        init_A = torch.zeros(self.A.shape, dtype=datatype, device=DEVICE)
        init_A.copy_(self.A)
        init_B = torch.zeros(self.B.shape, dtype=datatype, device=DEVICE)
        init_B.copy_(self.B)

        # A_old refer to the value of A in the last iteratio
        self.A_old = torch.zeros(self.A.shape, dtype=datatype, device=DEVICE)
        self.A_old.copy_(self.A)
        self.B_old = torch.zeros(self.B.shape, dtype=datatype, device=DEVICE)
        self.B_old.copy_(self.B)
        self.U_old = torch.zeros(self.U.shape, dtype=datatype, device=DEVICE)
        self.U_old.copy_(self.U)
        self.V_old = torch.zeros(self.V.shape, dtype=datatype, device=DEVICE)
        self.V_old.copy_(self.V)

        A_best = torch.zeros(self.A.shape, dtype=datatype, device=DEVICE)
        B_best = torch.zeros(self.B.shape, dtype=datatype, device=DEVICE)
        U_best = torch.zeros(self.U.shape, dtype=datatype, device=DEVICE)
        V_best = torch.zeros(self.V.shape, dtype=datatype, device=DEVICE)

        # iteration
        while numiter > 0:
            Wlist = [W_all]

            # use a list of different mask to resample pos samples
            for num1 in range(self.resample):
                Wlist.append(W_all * Y)  # Pos sample

            for epoch in range(len(Wlist)):
                W_ = Wlist[epoch]
                Y_p = mm(self.A, mm(self.U, mm(self.V.t(), self.B)))
                P = torch.sigmoid(Y_p)

                # update U,V
                t1 = time()
                # reinitialize the optimizer
                opter_U = opter(lr=lr, shape=self.U.shape, DEVICE=DEVICE)
                opter_V = opter(lr=lr, shape=self.V.shape, DEVICE=DEVICE)
                for foo in range(30):
                    # compute the derivative of U and V
                    deriv_U = (mm(mm(self.A.t(), P * W_), mm(self.B.t(), self.V))
                               + mm(mm((c - 1.) * self.A.t(), Y * P * W_), mm(self.B.t(), self.V))
                               - c * mm(mm(self.A.t(), Y * W_), mm(self.B.t(), self.V))
                               + 2. * lamb * self.U
                               + 2. * r1 * mm(mm(self.A.t(), self.lap_mat(self.drugMat)), mm(self.A, self.U)))
                    deriv_V = (mm(self.B, mm((P * W_).t(), mm(self.A, self.U)))
                               + (c - 1.) * mm(self.B, mm((Y * P * W_).t(), mm(self.A, self.U)))
                               - c * mm(self.B, mm((Y * W_).t(), mm(self.A, self.U)))
                               + 2. * lamb * self.V
                               + 2. * r2 * mm(self.B, mm(self.lap_mat(self.targetMat), mm(self.B.t(), self.V))))

                    # update using adam optimizer
                    self.U += opter_U.delta(deriv_U, max_iter - numiter)
                    self.V += opter_V.delta(deriv_V, max_iter - numiter)
                    Y_p = mm(self.A, mm(self.U, mm(self.V.t(), self.B)))
                    P = torch.sigmoid(Y_p)

                    # break the loop if reach converge condition
                    if ((torch.norm(self.U - self.U_old, p=1) / torch.norm(self.U_old, p=1))
                        < 0.01) and ((torch.norm(self.V - self.V_old, p=1) / torch.norm(self.V_old, p=1)) < 0.01):
                        logging.debug("UV update num: %d" % foo)
                        break
                    self.U_old.copy_(self.U)
                    self.V_old.copy_(self.V)

                # store matrix U, V, A and B for the currently lowest loss for later use
                self.loss[self.n].append(loss_function(Y, W_all))
                if self.loss[self.n][-1] < minloss:
                    A_best.copy_(self.A)
                    B_best.copy_(self.B)
                    U_best.copy_(self.U)
                    V_best.copy_(self.V)
                    minloss = self.loss[self.n][-1]

                # Output the present performance evaluation
                if not (W == 1).all():
                    y_true = intMat[W == 0].flatten()
                    y_pre = P.cpu().numpy()[W == 0].flatten()
                    fpr, tpr, thresholds = roc_curve(y_true, y_pre)
                    p, r, thresholds = precision_recall_curve(y_true, y_pre)
                    auc = AUC(fpr, tpr)
                    aupr = AUC(r, p)
                    logging.info(
                        ('Dataset[%d]: Iteration[U, V] %d: AUC = %.4f, AUPR = %.4f, loss = %.4f, lr = %.4f [%.1fs]'
                         % (self.n, int(max_iter - numiter), auc, aupr, self.loss[self.n][-1], lr, time() - t1)))
                else:
                    logging.info(('Dataset[%d]: Iteration[U, V] %d: loss = %.4f, lr = %.4f [%.1fs]' % (
                        self.n, int(max_iter - numiter), self.loss[self.n][-1], lr, time() - t1)))

                # if diverge reinitialize U, V, A, B and optimizer with half the present learning rate
                if self.loss[self.n][-1] > self.loss[self.n][1] * 2:
                    if not patient:
                        with open('./error.log', 'a') as error:
                            err = 'Converge error\n' + str(self) + '\t' + asctime() + '\n'
                            err += 'Current lr: %f\n' % lr
                            error.write(err)
                        self.A = A_best
                        self.B = B_best
                        self.U = U_best
                        self.V = V_best
                        logging.error('Dataset[%d], U&V: Fail to reach convergence, exit with best loss=%.2f' %
                                      (self.n, min(self.loss[self.n])))
                        return
                    # Reinitialization
                    self.A.copy_(init_A)
                    self.B.copy_(init_B)
                    self.U.copy_(init_U)
                    self.V.copy_(init_V)
                    lr = lr * 0.5
                    numiter.copy_(max_iter)
                    self.loss[self.n][-1] = np.inf
                    patient -= 1
                    logging.warning("Dataset[%d], U&V: Diverged, attempt to retrain with half the present "
                                    "learning rate, lr=%.4f/2, patient=%d-1, "
                                    "best loss=%.2f" % (self.n, lr * 2, patient + 1, min(self.loss[self.n])))
                    break

                # Update A & B
                t1 = time()
                for n in range(30):
                    temp_p = ((mm(self.U, mm(self.V.t(), self.B))).t() + torch.abs(
                        mm(self.U, mm(self.V.t(), self.B)).t())) * 0.5
                    temp_n = (torch.abs(mm(self.U, mm(self.V.t(), self.B)).t())
                              - mm(self.U, mm(self.V.t(), self.B)).t()) * 0.5
                    UUT_p = (mm(self.U, self.U.t()) + torch.abs(mm(self.U, self.U.t()))) * 0.5
                    UUT_n = (torch.abs(mm(self.U, self.U.t())) - mm(self.U, self.U.t())) * 0.5
                    D_AP = (mm(P * W_, temp_p) + (c - 1) * mm(Y * P * W_, temp_p)
                            + c * mm(Y * W_, temp_n) + beta * (2 * self.A)
                            + 2. * r1 * mm(torch.diag(self.drugMat.sum(1)), mm(self.A, UUT_p))
                            + 2. * r1 * mm(self.drugMat, mm(self.A, UUT_n)))
                    D_AN = (mm(P * W_, temp_n) + (c - 1) * mm(Y * P * W_, temp_n)
                            + c * mm(Y * W_, temp_p)
                            + 2. * r1 * mm(torch.diag(self.drugMat.sum(1)), mm(self.A, UUT_n))
                            + 2. * r1 * mm(self.drugMat, mm(self.A, UUT_p)))
                    temp_p = ((mm(self.A, mm(self.U, self.V.t()))).t() + torch.abs(
                        (mm(self.A, mm(self.U, self.V.t()))).t())) * 0.5
                    temp_n = (torch.abs(mm(self.A, mm(self.U, self.V.t())).t())
                              - (mm(self.A, mm(self.U, self.V.t()))).t()) * 0.5
                    VVT_p = (mm(self.V, self.V.t()) + torch.abs(mm(self.V, self.V.t()))) * 0.5
                    VVT_n = (torch.abs(mm(self.V, self.V.t())) - mm(self.V, self.V.t())) * 0.5

                    D_BP = (mm(temp_p, P * W_) + (c - 1) * mm(temp_p, Y * P * W_)
                            + c * mm(temp_n, Y * W_)
                            + beta * (2 * self.B)
                            + 2. * r2 * mm(VVT_p, mm(self.B, torch.diag(self.targetMat.sum(1))))
                            + 2. * r2 * mm(VVT_n, mm(self.B, self.targetMat)))
                    D_BN = (mm(temp_n, P * W_) + (c - 1) * mm(temp_n, Y * P * W_)
                            + c * mm(temp_p, Y * W_)
                            + 2. * r2 * mm(VVT_n, mm(self.B, torch.diag(self.targetMat.sum(1))))
                            + 2. * r2 * mm(VVT_p, mm(self.B, self.targetMat)))
                    temp = (self.A * (1. / (D_AP + eps))).sum(1).flatten()
                    D_SA = torch.diag(temp)  # refer to D superscript A
                    E_SA = (self.A * D_AN * (1. / (D_AP + eps))).sum(1).reshape(self.num_drugs, 1).repeat(1,
                                                                                                          self.num_drugs)
                    temp = (self.B * (1. / (D_BP + eps))).sum(0).flatten()
                    D_SB = torch.diag(temp)  # refer to D superscript B
                    E_SB = (self.B * D_BN * (1. / (D_BP + eps))).sum(0).reshape(1, self.num_targets).repeat(
                        self.num_targets, 1)

                    self.A = self.A * (mm(D_SA, D_AN) + 1) * (1. / (mm(D_SA, D_AP) + E_SA + eps))
                    self.B = self.B * (mm(D_BN, D_SB) + 1) * (1. / (mm(D_BP, D_SB) + E_SB + eps))
                    Y_p = mm(self.A, mm(self.U, mm(self.V.t(), self.B)))
                    P = torch.sigmoid(Y_p)

                    # break the loop if reach converge condition
                    if ((torch.norm(self.A - self.A_old, p=1) / torch.norm(self.A_old, p=1))
                        < 0.01) and ((torch.norm(self.B - self.B_old, p=1) / torch.norm(self.B_old, p=1)) < 0.01):
                        logging.debug("AB update num: %d" % n)
                        break
                    self.A_old.copy_(self.A)
                    self.B_old.copy_(self.B)

                # store matrix U, V, A and B for the currently lowest loss for later use
                self.loss[self.n].append(loss_function(Y, W_all))
                if self.loss[self.n][-1] < minloss:
                    A_best.copy_(self.A)
                    B_best.copy_(self.B)
                    U_best.copy_(self.U)
                    V_best.copy_(self.V)
                    minloss = self.loss[self.n][-1]

                # Output the present performance evaluation
                if not (W == 1).all():
                    y_true = intMat[W == 0].flatten()
                    y_pre = P.cpu().numpy()[W == 0].flatten()
                    fpr, tpr, thresholds = roc_curve(y_true, y_pre)
                    p, r, thresholds = precision_recall_curve(y_true, y_pre)
                    auc = AUC(fpr, tpr)
                    aupr = AUC(r, p)
                    logging.info(('Dataset[%d]: Iteration[A, B] %d: AUC = %.4f, AUPR = %.4f, loss = %.4f [%.1fs]' % (
                        self.n, int(max_iter - numiter), auc, aupr, self.loss[self.n][-1], time() - t1)))
                else:
                    logging.info(('Dataset[%d]: Iteration[A, B] %d: loss = %.4f [%.1fs]' % (
                        self.n, int(max_iter - numiter), self.loss[self.n][-1], time() - t1)))

                # if diverge reinitialize U, V, A, B and optimizer with half the present learning rate
                if self.loss[self.n][-1] > self.loss[self.n][1] * 2:
                    if not patient:
                        with open('./error.log', 'a') as error:
                            err = 'early termination: Do not converge\n' + str(self) + '\t' + asctime() + '\n'
                            err += 'Current lr: %f\n' % lr
                            error.write(err)
                        self.A = A_best
                        self.B = B_best
                        self.U = U_best
                        self.V = V_best
                        logging.error('Dataset[%d], A&B: Fail to reach convergence, exit with best loss=%.2f' %
                                      (self.n, min(self.loss[self.n])))
                        return
                    # Reinitialization
                    self.A.copy_(init_A)
                    self.B.copy_(init_B)
                    self.U.copy_(init_U)
                    self.V.copy_(init_V)
                    lr = lr * 0.5
                    numiter.copy_(max_iter)
                    self.loss[self.n][-1] = np.inf
                    patient -= 1
                    logging.warning("Dataset[%d], A&B: Diverged, attempt to retrain with half the present "
                                    "learning rate, lr=%.4f/2, patient=%d-1, "
                                    "best loss=%.2f" % (self.n, lr * 2, patient + 1, min(self.loss[self.n])))
                    break
                else:
                    # training stop if reach converge condition
                    delta_loss = abs(self.loss[self.n][-1] - self.loss[self.n][-2]) / abs(self.loss[self.n][-2])
                    logging.info(('Delta_loss: %.4f' % delta_loss))
                    if delta_loss < 1e-4:
                        numiter = torch.tensor(data=0, dtype=torch.int, device=DEVICE)
            numiter -= 1

        # retrieve the best U, V, A and B (with lowest loss)
        if self.loss[self.n][-1] > minloss:
            self.A = A_best
            self.B = B_best
            self.U = U_best
            self.V = V_best

    def evaluation(self, test_data, test_label, verpose=1):
        '''Evaluation

        Args:
            test_data: testing set of data
            test_label: testing set of label
            verpose:

        Returns:
            aupr, auc
        '''
        t1 = time()
        # Evaluation
        Y_p = mm(self.A, mm(self.U, mm(self.V.t(), self.B)))
        P = torch.sigmoid(Y_p).cpu().numpy()
        test_data = test_data.T
        y_pre = P[test_data[0], test_data[1]]

        fpr, tpr, thresholds = roc_curve(test_label, y_pre)
        p, r, thresholds = precision_recall_curve(test_label, y_pre)
        auc_val = AUC(fpr, tpr)
        aupr_val = AUC(r, p)
        logging.info(('Dataset[%d]: Iteration %d : AUC = %.4f, AUPR = %.4f, [%.1f s]'
                      % (self.n, self.max_iter, auc_val, aupr_val, time() - t1)))
        return aupr_val, auc_val

    def test(self, test_data):
        '''return the predicting label of input data

        Args:
            test_data: input data

        Returns:
            predicting label
        '''

        t1 = time()
        # Evaluation
        Y_p = mm(self.A, mm(self.U, mm(self.V.t(), self.B)))
        P = torch.sigmoid(Y_p).cpu().numpy()
        test_data = test_data.T
        y_pre = P[test_data[0], test_data[1]]
        return y_pre

    def predict_scores(self, test_data, N):
        '''return the predicting label of input data

        Args:
            test_data: input data

        Returns:
            predicting label
        '''

        # Evaluation
        test_data = np.array(test_data)
        Y_p = mm(self.A, mm(self.U, mm(self.V.t(), self.B)))
        P = torch.sigmoid(Y_p).cpu().numpy()
        test_data = test_data.T
        y_pre = P[test_data[0], test_data[1]]
        return y_pre

    def lap_mat(self, S):
        '''Return the Laplacian matrix of adjacent matrix S

        Args:
            S: adjacent matrix

        Returns:
            Laplacian matrix of S
        '''
        x = S.sum(1)
        L = torch.diag(x) - S  # neighborhood regularization matrix
        return L

    class adam_opt:
        def __init__(self, lr, shape, DEVICE):
            '''Adam optimizer

            Args:
                lr:
                shape:
            '''
            self.alpha = torch.tensor(data=lr, dtype=datatype, device=DEVICE)
            self.beta1 = torch.tensor(data=0.9, dtype=datatype, device=DEVICE)
            self.beta2 = torch.tensor(data=0.999, dtype=datatype, device=DEVICE)
            self.epsilon = torch.tensor(data=10E-8, dtype=datatype, device=DEVICE)
            self.eps = torch.tensor(data=npeps, dtype=datatype, device=DEVICE)
            self.t = torch.tensor(data=0, dtype=datatype, device=DEVICE)
            self.m0 = torch.zeros(shape, dtype=datatype, device=DEVICE)
            self.v0 = torch.zeros(shape, dtype=datatype, device=DEVICE)

        def delta(self, deriv, iter):
            # in case pass a matrix type grad
            self.t = (iter + 1).type(datatype)
            grad = deriv
            m_t = self.beta1 * self.m0 + (1 - self.beta1) * grad
            v_t = self.beta2 * self.v0 + (1 - self.beta2) * grad ** 2
            # In this project the number of iteration is too big so let t divided by a number
            m_cap = m_t / (1. - self.beta1 ** (self.t / 1.) + self.eps)
            v_cap = v_t / (1. - self.beta2 ** (self.t / 1.) + self.eps)
            update = - self.alpha * m_cap / (torch.sqrt(v_cap) + self.epsilon + self.eps)
            self.m0.copy_(m_t)
            self.v0.copy_(v_t)
            return update

    def __str__(self):
        return ("Model: ADPGMF, cvs: %s, K: %s, mf_dim: %s, lamb:%s, beta:%s, c:%s, resample:%d, r1:%s, r2:%s, ita:%s "
                "imp1:%s, imp2: %s, max_iter: %s, lr: %s" % (self.cvs, self.K, self.mf_dim, self.lamb, self.beta,
                                                             self.c, self.resample, self.r1, self.r2, self.ita,
                                                             self.imp1, self.imp2, self.max_iter, self.lr))
