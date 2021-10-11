import numpy as np
cimport numpy as np
from cpython cimport array
import array
DTYPE = np.double
np.import_array()
ctypedef np.double_t DTYPE_t #compile-time type
ctypedef np.float64_t ADP_DTYPE_t
ctypedef np.int_t DTYPE_INT_t

import sys
sys.path.append('../../')
from numpy import multiply,  dot
from numpy.matlib import repmat
from numpy.linalg import norm
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc as AUC
from time import time, asctime
npeps = np.finfo(float).eps

cdef list adpgmf_iter(list Wlist, np.ndarray[ADP_DTYPE_t, ndim=2] Y, np.ndarray[ADP_DTYPE_t, ndim=2] drugMat,
                      np.ndarray[ADP_DTYPE_t, ndim=2] targetMat, dict args, int seed, int n):
    cdef list loss = []
    cdef int num_drugs = Y.shape[0], num_targets = Y.shape[1]
    cdef int num_factors = args['mf_dim'], c = args['c'], max_iter = args['max_iter'], verpose=args['verpose']
    cdef int patient = args['patient'], numiter = max_iter
    cdef float lamb = args['lamb'], beta = args['beta'], r1 = args['r1'], r2 = args['r2'], lr = args['lr']
    cdef float t1
    cdef str cmd = args['cmd']
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] A = dot(np.diag(1. / np.sum(drugMat + npeps, axis=1).flatten()), drugMat + npeps)
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] B = dot(targetMat + npeps, np.diag(1. / np.sum(targetMat + npeps, axis=0).flatten()))
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] U = np.sqrt(1. / float(num_factors)) * np.random.normal(size=(num_drugs, num_factors))
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] V = np.sqrt(1. / float(num_factors)) * np.random.normal(size=(num_targets, num_factors))
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] init_A = A.copy(), init_B = B.copy(), init_U, init_V
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] old_A = A.copy(), old_B = B.copy(), old_U = U.copy(), old_V = V.copy()
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] W_all = Wlist[0].copy()
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] intMat1 = Y * W_all
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] DL = lap_mat(drugMat), TL = lap_mat(targetMat)
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] Y_p, P, W, deriv_U, deriv_V, temp_p, temp_n, D_AP, D_AN, D_SA, E_SA
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] D_BP, D_BN, D_SB, E_SB
    cdef np.ndarray[ADP_DTYPE_t, ndim=1] temp
    cdef list U_shape = [len(U), len(U.T)], V_shape = [len(V), len(V.T)]
    cdef adam_opt opter_U = adam_opt(lr=lr, shape=U_shape) , opter_V = adam_opt(lr=lr, shape=(len(V), len(V.T)))
    prng = np.random.RandomState(seed)
    U = np.sqrt(1. / float(num_factors)) * prng.normal(size=(num_drugs, num_factors))
    V = np.sqrt(1. / float(num_factors)) * prng.normal(size=(num_targets, num_factors))
    init_U = U.copy()
    init_V = V.copy()
    while numiter:
        for epoch in range(len(Wlist)):
            W = Wlist[epoch]
            # for epoch in range(1):
            t0 = time()
            Y_p = dot(A, dot(U, dot(V.T, B)))
            P = sigmoid(Y_p)
            # update u,v and a,b respectively
            t1 = time()
            for foo in range(5):
                deriv_U = (dot(A.T, dot((P+(c-1)*Y*P-c*Y)*W, dot(B.T, V)))
                           + 2. * lamb * U
                           + 2. * r1 * dot(dot(A.T, DL), dot(A, U)))

                deriv_V = (dot(B, dot((P.T + (c - 1) * (Y * P).T
                                            - c * Y.T) * W.T, dot(A, U)))
                           + 2. * lamb * V
                           + 2. * r2 * dot(B, dot(TL, dot(B.T, V))))

                # update using adam optimizer
                U += opter_U.delta(deriv_U, max_iter - numiter)
                V += opter_V.delta(deriv_V, max_iter - numiter)
                Y_p = dot(A, dot(U, dot(V.T, B)))
                P = sigmoid(Y_p)
            loss.append(loss_function(Y, W,A, B, U, V, drugMat, targetMat, lamb, beta, r1, r2, c))
            if loss[-1] < minloss:
                A_best, B_best, U_best, V_best = A.copy(), B.copy(), U.copy(), V.copy()
                minloss = loss[-1]
            if not ((numiter * (epoch + 1)) % verpose):
                y_true = Y[W_all == 0].flatten()
                y_pre = P[W_all == 0].flatten()
                fpr, tpr, thresholds = roc_curve(y_true, y_pre)
                p, r, thresholds = precision_recall_curve(y_true, y_pre)
                auc = AUC(fpr, tpr)
                aupr = AUC(r, p)
                print(('Dataset[%d]: Iteration[U, V] %d: AUC = %.4f, AUPR = %.4f, loss = %.4f, lr = %.4f [%.1fs]' % (
                    n, max_iter - numiter, auc, aupr, loss[-1], lr, time() - t1)))

            if loss[-1] > loss[1] * 10:
                if not patient:
                    with open('./error.log', 'a') as error:
                        err = 'Converge error\n' + cmd + '\t' + asctime() + '\n'
                        err += 'Current lr: %f\n' % lr
                        error.write(err)
                    A = A_best.copy()
                    B = B_best.copy()
                    U = U_best.copy()
                    V = V_best.copy()
                    return

                # Reinitialization
                A = init_A.copy()
                B = init_B.copy()
                U = init_U.copy()
                V = init_V.copy()
                lr = lr * 0.5
                opter_U = adam_opt(lr=lr, shape=U_shape)
                opter_V = adam_opt(lr=lr, shape=V_shape)
                numiter = max_iter
                loss[-1] = np.inf
                patient -= 1
                break

            old_A = A.copy()
            old_B = B.copy()
            old_U = U.copy()
            old_V = V.copy()

            # Update A & B
            t1 = time()
            temp_p = ((dot(U, dot(V.T, B))).T + np.abs(dot(U, dot(V.T, B)).T)) * 0.5
            temp_n = (np.abs(dot(U, dot(V.T, B)).T) - dot(U, dot(V.T, B)).T) * 0.5
            D_AP = (dot((1 + (c - 1) * Y)*W*P, temp_p) + c * dot(intMat1, temp_n)
                    + beta * (2 * A)
                    + 2. * r1 * dot(np.diag(np.sum(drugMat, 1)), dot(A, dot(U, U.T))))
            D_AN = (dot((1+ (c - 1) * Y) * W * P, temp_n) + c * dot(intMat1, temp_p)
                    + 2. * r1 * dot(drugMat, dot(A, dot(U, U.T))))
            temp = (np.sum(A * (1. / (D_AP + npeps)), axis=1)).flatten()
            D_SA = np.diag(temp)  # refer to D superscript A
            E_SA = repmat(np.sum(A * D_AN * (1. / (D_AP + npeps)), axis=1).reshape(num_drugs, 1), 1,
                          num_drugs)
            A = A * ((dot(D_SA, D_AN) + 1) * (1. / (dot(D_SA, D_AP) + E_SA + npeps)))
            Y_p = dot(A, dot(U, dot(V.T, B)))
            P = sigmoid(Y_p)

            temp_p = ((dot(A, dot(U, V.T))).T + np.abs((dot(A, dot(U, V.T))).T)) * 0.5
            temp_n = (np.abs((dot(A, dot(U, V.T))).T) - (dot(A, dot(U, V.T))).T) * 0.5
            D_BP = (dot(temp_p, (1 + (c - 1) * Y) * P * W) + c * dot(temp_n, intMat1)
                    + beta * (2 * B)
                    + 2. * r2 * dot(V, dot(V.T, dot(B, np.diag(np.sum(targetMat, 1))))))
            D_BN = (dot(temp_n, (1 + (c - 1) * Y) * P * W) + c * dot(temp_p, intMat1)
                    + 2. * r2 * dot(V, dot(V.T, dot(B, targetMat))))
            temp = (np.sum(B * (1. / (D_BP + npeps)), axis=0)).flatten()
            D_SB = np.diag(temp)  # refer to D superscript B
            E_SB = repmat(np.sum(B * D_BN * (1. / (D_BP + npeps)), axis=0).reshape(1, num_targets),
                          num_targets, 1)
            B = B * (dot(D_BN, D_SB) + 1) * (1. / (dot(D_BP, D_SB) + E_SB + npeps))
            Y_p = dot(A, dot(U, dot(V.T, B)))
            P = sigmoid(Y_p)

            loss.append(loss_function(Y, W,A, B, U, V, drugMat, targetMat, lamb, beta, r1, r2, c))
            if loss[-1] < minloss:
                A_best, B_best, U_best, V_best = A.copy(), B.copy(), U.copy(), V.copy()
                minloss = loss[-1]

            if not ((numiter * (epoch + 1)) % verpose):
                y_true = Y[W_all == 0].flatten()
                y_pre = P[W_all == 0].flatten()
                fpr, tpr, thresholds = roc_curve(y_true, y_pre)
                p, r, thresholds = precision_recall_curve(y_true, y_pre)
                auc = AUC(fpr, tpr)
                aupr = AUC(r, p)
                print(('Dataset[%d]: Iteration[A, B] %d: AUC = %.4f, AUPR = %.4f, loss = %.4f [%.1fs]' % (
                    n, max_iter - numiter, auc, aupr, loss[-1], time() - t1)))

                delta_loss = abs(loss[-1] - loss[-2]) / abs(loss[-2])
                delta_U = norm(U - old_U) / norm(old_U)
                delta_V = norm(V - old_V) / norm(old_V)
                delta_A = norm(A - old_A) / norm(old_A)
                delta_B = norm(B - old_B) / norm(old_B)
                print(('Delta_loss: %.4f, delta_U: %.4f, delta_V: %.4f, delta_A: %.4f, delta_B:%.4f'
                      % (delta_loss, delta_U, delta_V, delta_A, delta_B)))
            elif (max_iter - numiter)  % 10 == 0:
                print(('Iteration: %d, loss: %.4f, lr: %.4f, t: %.2f' % (
                    max_iter - numiter, loss[-1], lr,
                    time() - tt)))
                tt = time()

            # Ealry terminated, let lr = lr*0.5 and try again.
            if loss[-1] > loss[1] * 10:
                if not patient:
                    with open('./error.log', 'a') as error:
                        err = 'early termination: Do not converge\n' + cmd + '\t' + asctime() + '\n'
                        err += 'Current lr: %f\n' % lr
                        error.write(err)
                    A = A_best.copy()
                    B = B_best.copy()
                    U = U_best.copy()
                    V = V_best.copy()
                    return

                # Reinitialization
                A = init_A.copy()
                B = init_B.copy()
                U = init_U.copy()
                V = init_V.copy()

                lr = lr * 0.5
                opter_U = adam_opt(lr=lr, shape=U_shape)
                opter_V = adam_opt(lr=lr, shape=V_shape)
                numiter = max_iter
                loss[-1] = np.inf
                patient -= 1
                break
            old_A = A.copy()
            old_B = B.copy()
            old_U = U.copy()
            old_V = V.copy()
        numiter -= 1
    return [A, B, U, V, loss]


cdef np.ndarray[ADP_DTYPE_t, ndim=2] sigmoid(np.ndarray[ADP_DTYPE_t, ndim=2] x):
    cdef list shape=[x.shape[0], x.shape[1]]
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] re = np.zeros(shape)
    re[x >= -100] = 1. / (1 + np.exp(-x[x >= -100]))
    return re

cdef np.ndarray[ADP_DTYPE_t, ndim=2] lap_mat(np.ndarray[ADP_DTYPE_t, ndim=2] S):
    cdef np.ndarray[ADP_DTYPE_t, ndim=1] x = np.sum(S, axis=1)  # TODO:col? or row
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] L = np.diag(x) - S  # neighborhood regularization matrix
    return L

cdef np.ndarray[ADP_DTYPE_t, ndim=2] loss_function(np.ndarray[ADP_DTYPE_t, ndim=2] Y, np.ndarray[ADP_DTYPE_t, ndim=2] W,
                                                   np.ndarray[ADP_DTYPE_t, ndim=2] A, np.ndarray[ADP_DTYPE_t, ndim=2] B,
                                                   np.ndarray[ADP_DTYPE_t, ndim=2] U, np.ndarray[ADP_DTYPE_t, ndim=2] V,
                                                   np.ndarray[ADP_DTYPE_t, ndim=2] drugMat,
                                                   np.ndarray[ADP_DTYPE_t, ndim=2] targetMat,
                                                   float lamb, float beta, float r1, float r2, int c):
        cdef np.ndarray[ADP_DTYPE_t, ndim=2] temp = dot(A, dot(U, dot(V.T, B)))
        cdef np.ndarray[ADP_DTYPE_t, ndim=2] logexp = np.zeros([len(temp), len(temp.T)])
        # temp = dot(A, dot(U, dot(V.T, B)))  # avoiding overflow
        # logexp = np.zeros(temp.shape)
        logexp[temp > 50] = temp[temp > 50] * np.log(np.e)
        logexp[temp <= 50] = np.log(np.exp(temp[temp <= 50]) + 1)

        loss = (multiply(multiply((1 + c * Y - Y), logexp)
                         - multiply(c * Y, dot(A, dot(U, dot(V.T, B)))), W).sum()
                + lamb * (norm(U, ord='fro') ** 2 + norm(V, ord='fro') ** 2)
                + beta * (norm(A, ord='fro') ** 2 + norm(B, ord='fro') ** 2)
                + r1 * np.trace(dot(U.T, dot(A.T, dot(lap_mat(drugMat), dot(A, U)))))
                + r2 * np.trace(dot(V.T, dot(B, dot(lap_mat(targetMat), dot(B.T, V))))))
        return loss

cdef class adam_opt:
    def __init__(self, lr, shape):
        self.alpha = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 10E-8
        self.t = 0
        self.m0 = np.zeros(shape)
        self.v0 = np.zeros(shape)

    cdef delta(self, np.ndarray[ADP_DTYPE_t, ndim=2] deriv, int iter):
        self.t = iter + 1
        npeps = np.finfo(float).eps
        grad = deriv.copy()
        m_t = self.beta1 * self.m0 + (1 - self.beta1) * grad
        v_t = self.beta2 * self.v0 + (1 - self.beta2) * grad ** 2
        # In this project the number of iteration is too big so let t divided by a number
        m_cap = m_t / (1. - self.beta1 ** (self.t / 1.) + npeps)
        v_cap = v_t / (1. - self.beta2 ** (self.t / 1.) + npeps)
        update = - self.alpha * m_cap / (np.sqrt(v_cap) + self.epsilon + npeps)
        self.m0 = m_t.copy()
        self.v0 = v_t.copy()
        return update