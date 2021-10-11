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
from numpy.matlib import repmat
from numpy.linalg import norm
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc as AUC
from time import asctime
npeps = np.finfo(float).eps

#time
from libc.time cimport time,time_t

# cdef time_t t = time(NULL)




# cimport cython
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef pbmda_fix(np.ndarray[DTYPE_t, ndim=2] Y, np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B, double Similarity_threshold, double Pa, int Max_length):
    assert Y.dtype == DTYPE and A.dtype == DTYPE and B.dtype == DTYPE
    cdef int i, j, k, m, q, index, lvalue=0
    cdef DTYPE_t temp
    cdef int y = Y.shape[0]  # y: num of miRNAs, x: num of diseases
    cdef int x = Y.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] P = np.zeros([x, y], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] Network_disease = np.zeros([x, 4, max(x, y)], dtype=DTYPE)  # 0:disease 1:miRNA
    cdef np.ndarray[DTYPE_t, ndim=3] Network_miRNA = np.zeros([y, 4, max(x, y)], dtype=DTYPE)
    cdef list l = []
    cdef list ll = []
    cdef list lll = []
    cdef list ll1 = []
    # cdef array.array lll = array.array('l',[])
    cdef array.array trick = array.array('l',[])
    cdef array.array trick1 = array.array('l',[])
    cdef array.array trick2 = array.array('l',[])
    # constructed a heterogeneous graph consisting of three interlinked sub-graphs


    for i in range(x):
        index = 0
        for j in range(x):
            if B[i,j] > Similarity_threshold:
                Network_disease[i,0,index] = B[i,j]  # disease-disease  similarity
                Network_disease[i,1,index] = (j + 1)
                index += 1
        index = 0
        for k in range(y):
            if Y[k,i] == 1:
                Network_disease[i,2,index] = 1  # disease-miRNA similarity
                Network_disease[i,3,index] = (k + 1)
                index += 1

    for i in range(y):
        index = 0
        for j in range(y):
            if A[i, j] > Similarity_threshold:
                Network_miRNA[i,2,index] = A[i, j]  # miRNA-miRNA  similarity
                Network_miRNA[i,3,index] = (j + 1)
                index += 1
        index = 0
        for k in range(x):
            if Y[i, k] == 1:
                Network_miRNA[i,0,index] = 1  # disease-disease similarity
                Network_miRNA[i,1,index] = (k + 1)
                index += 1
                ####################################################################

    ####################################################################
    # It is a little complicated to explain the process.
    # For example, this part works for the first iteration

    # for i in range(x):  #+:disease -:miRNA
    for i in range(x):  # +:disease -:miRNA
        # print(i)
        # l = []
        # ll = []
        # lll = [i + 1]
        l.clear()
        ll.clear()
        lll.clear()
        lll.append(i+1)

        for k in range(max(y, x)):  # disease
            if Network_disease[i,0,k] == 0:
                break
            if k != i:
                lll.append(<int>(Network_disease[i,1,k]))
                ll.append(lll)
                lll = [i + 1]
        for k in range(max(y, x)):  # miRNA
            if Network_disease[i,2,k] == 0:
                break
            lll.append(-<int>(Network_disease[i,3,k]))
            ll.append(lll)
            lll = [i + 1]
            P[i, -<int>(ll[-1][-1]) - 1] += 1

        ####################################################################

        ####################################################################
        # This part worked for the rest iterations based on the selected maximun path length
        # for j in range(Max_length-1):



        for j in range(Max_length - 1):
            ll1 = []
            # ll1.clear()
            for k in range(len(ll)):
                # unit=ll[k]
                # lll = list(ll[k])
                trick = array.array('l',ll[k])
                len_trick = len(trick)

                if trick.data.as_ints[j + 1] > 0:  # disease
                    for m in range(max(y, x)):  # disease
                        if Network_disease[trick.data.as_ints[j + 1] - 1, 0, m] == 0:
                            break;
                        # if (Network_disease[<int>(trick.data.as_ints[j + 1] - 1), 1, m]) not in trick:
                        #     ll1.append(trick + array.array('l', [<int>Network_disease[<int>(trick.data.as_ints[j + 1] - 1), 1, m]]))
                        if not int_in(<int>Network_disease[trick.data.as_ints[j + 1] - 1, 1, m], trick, len_trick):
                            array.resize_smart(trick2, len_trick + 1)
                            copy_and_extend(trick2, trick, <int>Network_disease[trick.data.as_ints[j + 1] - 1, 1, m], len_trick + 1)
                            ll1.append(array.copy(trick2))

                            # ll1.append(trick + array.array('l', [<int>Network_disease[trick.data.as_ints[j + 1] - 1, 1, m]]))
                    for m in range(max(y, x)):  # miRNA
                        temp = 1
                        if Network_disease[trick.data.as_ints[j + 1] - 1, 2, m] == 0:
                            break;
                        # if (-(Network_disease[<int>(trick.data.as_ints[j + 1] - 1), 3, m])) not in trick: #TODO:?
                        #     ll1.append(trick + array.array('l', [-<int>Network_disease[<int>(trick.data.as_ints[j + 1] - 1), 3, m]]))

                        if not int_in(-<int>(Network_disease[trick.data.as_ints[j + 1] - 1, 3, m]), trick, len_trick):
                            array.resize_smart(trick1, len_trick + 1)
                            copy_and_extend(trick1, trick, -<int>Network_disease[trick.data.as_ints[j + 1] - 1, 3, m], len_trick + 1)
                            ll1.append(array.copy(trick1))

                            # ll1.append(trick + array.array('l', [-<int>Network_disease[trick.data.as_ints[j + 1] - 1, 3, m]]))
                            # trick1 = array.copy(ll1[-1])

                            # for q in range(j + 2):
                            #     if ll1[-1][q + 1] > 0 and ll1[-1][q] > 0:  # disease-disease
                            #         temp *= B[<int>(ll1[-1][q] - 1), <int>(ll1[-1][q + 1] - 1)]
                            #     elif ll1[-1][q + 1] < 0 and ll1[-1][q] > 0:  # disease-miRNA
                            #         temp *= 1
                            #     elif ll1[-1][q + 1] > 0 and ll1[-1][q] < 0:  # miRNA-disease
                            #         temp *= 1
                            #     elif ll1[-1][q + 1] < 0 and ll1[-1][q] < 0:  # miRNA-miRNA
                            #         temp *= A[-<int>(ll1[-1][q]) - 1, -<int>(ll1[-1][q + 1]) - 1]
                            # P[i, -<int>(ll1[-1][-1]) - 1] += (temp ** (Pa * (j + 2)))
                            for q in range(j + 2):
                                if trick1.data.as_ints[q + 1] > 0 and trick1.data.as_ints[q] > 0:  # disease-disease
                                    temp *= B[trick1.data.as_ints[q] - 1, trick1.data.as_ints[q + 1] - 1]
                                elif trick1.data.as_ints[q + 1] < 0 and trick1.data.as_ints[q] > 0:  # disease-miRNA
                                    temp *= 1
                                elif trick1.data.as_ints[q + 1] > 0 and trick1.data.as_ints[q] < 0:  # miRNA-disease
                                    temp *= 1
                                elif trick1.data.as_ints[q + 1] < 0 and trick1.data.as_ints[q] < 0:  # miRNA-miRNA
                                    temp *= A[-trick1.data.as_ints[q] - 1, -trick1.data.as_ints[q + 1] - 1]
                            P[i, -trick1.data.as_ints[len(trick1)-1] - 1] += (temp ** (Pa * (j + 2)))
                if trick.data.as_ints[j + 1] < 0:  # miRNA
                    for m in range(max(y, x)):  # disease
                        if Network_miRNA[-trick.data.as_ints[j + 1] - 1, 0, m] == 0:
                            break;
                        # if Network_miRNA[-<int>(trick.data.as_ints[j + 1]) - 1, 1, m] not in trick:
                        #     ll1.append(trick + array.array('l', [<int>Network_miRNA[-<int>(trick.data.as_ints[j + 1]) - 1, 1, m]]))
                        if not int_in(<int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 1, m], trick, len_trick):
                            array.resize_smart(trick2, len_trick + 1)
                            copy_and_extend(trick2, trick, <int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 1, m], len_trick + 1)
                            ll1.append(array.copy(trick2))

                            # ll1.append(trick + array.array('l', [<int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 1, m]]))
                    for m in range(max(y, x)):  # miRNA
                        temp = 1
                        if Network_miRNA[-trick.data.as_ints[j + 1] - 1, 2, m] == 0:
                            break;
                        # if (-Network_miRNA[-<int>(trick.data.as_ints[j + 1]) - 1, 3, m]) not in trick:
                        if not int_in(-<int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 3, m], trick, len_trick):
                            array.resize_smart(trick1, len_trick + 1)
                            copy_and_extend(trick1, trick, -<int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 3, m], len_trick + 1)
                            ll1.append(array.copy(trick1))

                            # ll1.append(trick + array.array('l', [-<int>Network_miRNA[-trick.data.as_ints[j + 1] - 1, 3, m]]))
                            # trick1 = array.copy(ll1[-1])
                            # for q in range(j + 2):
                            #     if ll1[-1][q + 1] > 0 and ll1[-1][q] > 0:  # disease-disease
                            #         temp *= B[<int>(ll1[-1][q] - 1), <int>(ll1[-1][q + 1] - 1)]
                            #     elif ll1[-1][q + 1] < 0 and ll1[-1][q] > 0:  # miRNA-disease
                            #         temp *= 1
                            #     elif ll1[-1][q + 1] > 0 and ll1[-1][q] < 0:  # miRNA-disease
                            #         temp *= 1
                            #     elif ll1[-1][q + 1] < 0 and ll1[-1][q] < 0:  # miRNA-miRNA
                            #         temp *= A[-<int>(ll1[-1][q]) - 1, -<int>(ll1[-1][q + 1]) - 1]
                            # # The end of each iterations need to be aggregated with the scores
                            # P[i][-<int>(ll1[-1][-1]) - 1] += (temp ** (Pa * (j + 2)))
                            for q in range(j + 2):
                                if trick1.data.as_ints[q + 1] > 0 and trick1.data.as_ints[q] > 0:  # disease-disease
                                    temp *= B[trick1.data.as_ints[q] - 1, trick1.data.as_ints[q + 1] - 1]
                                elif trick1.data.as_ints[q + 1] < 0 and trick1.data.as_ints[q] > 0:  # miRNA-disease
                                    temp *= 1
                                elif trick1.data.as_ints[q + 1] > 0 and trick1.data.as_ints[q] < 0:  # miRNA-disease
                                    temp *= 1
                                elif trick1.data.as_ints[q + 1] < 0 and trick1.data.as_ints[q] < 0:  # miRNA-miRNA
                                    temp *= A[-trick1.data.as_ints[q] - 1, -trick1.data.as_ints[q + 1] - 1]
                            # The end of each iterations need to be aggregated with the scores
                            P[i, -trick1.data.as_ints[len(trick1)-1] - 1] += (temp ** (Pa * (j + 2)))


            ll = ll1
    return P.T


cdef int int_in(int num, array.array A, int size):
    cdef int i
    for i in range(size):
        if A.data.as_ints[i] == num:
            return True
    return False

cdef void copy_and_extend(array.array self, array.array cparray, int exnum, int selflen):
    # assert len(self) = len(cparray) + 1
    cdef int i
    for i in range(selflen-1):
        self.data.as_ints[i] = cparray.data.as_ints[i]
    self.data.as_ints[selflen-1] = exnum


cpdef list adpgmf_iter(list Wlist, np.ndarray[ADP_DTYPE_t, ndim=2] Y, np.ndarray[ADP_DTYPE_t, ndim=2] drugMat,
                      np.ndarray[ADP_DTYPE_t, ndim=2] targetMat, dict args, int seed, int n):
    cdef list loss = [np.inf]
    cdef int num_drugs = Y.shape[0], num_targets = Y.shape[1]
    cdef int num_factors = args['mf_dim'], c = args['c'], max_iter = args['max_iter'], verpose=args['verpose']
    cdef int patient = args['patient'], numiter = max_iter
    cdef float lamb = args['lamb'], beta = args['beta'], r1 = args['r1'], r2 = args['r2'], lr = args['lr']
    cdef float minloss = np.inf
    cdef time_t tt = time(NULL), t1 = time(NULL)

    cdef str cmd = args['cmd']
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] A = np.dot(np.diag(1. / np.sum(drugMat + npeps, axis=1).flatten()), drugMat + npeps)
    cdef np.ndarray[ADP_DTYPE_t, ndim=2] B = np.dot(targetMat + npeps, np.diag(1. / np.sum(targetMat + npeps, axis=0).flatten()))
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
            Y_p = np.dot(A, np.dot(U, np.dot(V.T, B)))
            P = sigmoid(Y_p)
            # update u,v and a,b respectively
            t1 = time(NULL)
            for foo in range(5):
                deriv_U = (np.dot(A.T, np.dot((P+(c-1)*Y*P-c*Y)*W, np.dot(B.T, V)))
                           + 2. * lamb * U
                           + 2. * r1 * np.dot(np.dot(A.T, DL), np.dot(A, U)))

                deriv_V = (np.dot(B, np.dot((P.T + (c - 1) * (Y * P).T
                                            - c * Y.T) * W.T, np.dot(A, U)))
                           + 2. * lamb * V
                           + 2. * r2 * np.dot(B, np.dot(TL, np.dot(B.T, V))))

                # update using adam optimizer
                U += opter_U.delta(deriv_U, max_iter - numiter)
                V += opter_V.delta(deriv_V, max_iter - numiter)
                Y_p = np.dot(A, np.dot(U, np.dot(V.T, B)))
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
                    n, max_iter - numiter, auc, aupr, loss[-1], lr, time(NULL) - t1)))

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
            t1 = time(NULL)
            temp_p = ((np.dot(U, np.dot(V.T, B))).T + np.abs(np.dot(U, np.dot(V.T, B)).T)) * 0.5
            temp_n = (np.abs(np.dot(U, np.dot(V.T, B)).T) - np.dot(U, np.dot(V.T, B)).T) * 0.5
            D_AP = (np.dot((1 + (c - 1) * Y)*W*P, temp_p) + c * np.dot(intMat1, temp_n)
                    + beta * (2 * A)
                    + 2. * r1 * np.dot(np.diag(np.sum(drugMat, 1)), np.dot(A, np.dot(U, U.T))))
            D_AN = (np.dot((1+ (c - 1) * Y) * W * P, temp_n) + c * np.dot(intMat1, temp_p)
                    + 2. * r1 * np.dot(drugMat, np.dot(A, np.dot(U, U.T))))
            temp = (np.sum(A * (1. / (D_AP + npeps)), axis=1)).flatten()
            D_SA = np.diag(temp)  # refer to D superscript A
            E_SA = repmat(np.sum(A * D_AN * (1. / (D_AP + npeps)), axis=1).reshape(num_drugs, 1), 1,
                          num_drugs)
            A = A * ((np.dot(D_SA, D_AN) + 1) * (1. / (np.dot(D_SA, D_AP) + E_SA + npeps)))
            Y_p = np.dot(A, np.dot(U, np.dot(V.T, B)))
            P = sigmoid(Y_p)

            temp_p = ((np.dot(A, np.dot(U, V.T))).T + np.abs((np.dot(A, np.dot(U, V.T))).T)) * 0.5
            temp_n = (np.abs((np.dot(A, np.dot(U, V.T))).T) - (np.dot(A, np.dot(U, V.T))).T) * 0.5
            D_BP = (np.dot(temp_p, (1 + (c - 1) * Y) * P * W) + c * np.dot(temp_n, intMat1)
                    + beta * (2 * B)
                    + 2. * r2 * np.dot(V, np.dot(V.T, np.dot(B, np.diag(np.sum(targetMat, 1))))))
            D_BN = (np.dot(temp_n, (1 + (c - 1) * Y) * P * W) + c * np.dot(temp_p, intMat1)
                    + 2. * r2 * np.dot(V, np.dot(V.T, np.dot(B, targetMat))))
            temp = (np.sum(B * (1. / (D_BP + npeps)), axis=0)).flatten()
            D_SB = np.diag(temp)  # refer to D superscript B
            E_SB = repmat(np.sum(B * D_BN * (1. / (D_BP + npeps)), axis=0).reshape(1, num_targets),
                          num_targets, 1)
            B = B * (np.dot(D_BN, D_SB) + 1) * (1. / (np.dot(D_BP, D_SB) + E_SB + npeps))
            Y_p = np.dot(A, np.dot(U, np.dot(V.T, B)))
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
                    n, max_iter - numiter, auc, aupr, loss[-1], time(NULL) - t1)))

                delta_loss = abs(loss[-1] - loss[-2]) / abs(loss[-2])
                delta_U = norm(U - old_U) / norm(old_U)
                delta_V = norm(V - old_V) / norm(old_V)
                delta_A = norm(A - old_A) / norm(old_A)
                delta_B = norm(B - old_B) / norm(old_B)
                print(('Delta_loss: %.4f, delta_U: %.4f, delta_V: %.4f, delta_A: %.4f, delta_B:%.4f'
                      % (delta_loss, delta_U, delta_V, delta_A, delta_B)))
            elif (max_iter - numiter)  % 10 == 0:
                print(('Iteration: %d, loss: %.4f, lr: %.4f, t: %.2f' % (max_iter - numiter, loss[-1], lr, time(NULL) - tt)))
                tt = time(NULL)

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

cdef ADP_DTYPE_t loss_function(np.ndarray[ADP_DTYPE_t, ndim=2] Y, np.ndarray[ADP_DTYPE_t, ndim=2] W,
                                                   np.ndarray[ADP_DTYPE_t, ndim=2] A, np.ndarray[ADP_DTYPE_t, ndim=2] B,
                                                   np.ndarray[ADP_DTYPE_t, ndim=2] U, np.ndarray[ADP_DTYPE_t, ndim=2] V,
                                                   np.ndarray[ADP_DTYPE_t, ndim=2] drugMat,
                                                   np.ndarray[ADP_DTYPE_t, ndim=2] targetMat,
                                                   float lamb, float beta, float r1, float r2, int c):
        cdef np.ndarray[ADP_DTYPE_t, ndim=2] temp = np.dot(A, np.dot(U, np.dot(V.T, B)))
        cdef np.ndarray[ADP_DTYPE_t, ndim=2] logexp = np.zeros([len(temp), len(temp.T)])
        # temp = np.dot(A, np.dot(U, np.dot(V.T, B)))  # avoiding overflow
        # logexp = np.zeros(temp.shape)
        logexp[temp > 50] = temp[temp > 50] * np.log(np.e)
        logexp[temp <= 50] = np.log(np.exp(temp[temp <= 50]) + 1)

        loss = (np.np.multiply(np.np.multiply((1 + c * Y - Y), logexp)
                         - np.np.multiply(c * Y, np.dot(A, np.dot(U, np.dot(V.T, B)))), W).sum()
                + lamb * (norm(U, ord='fro') ** 2 + norm(V, ord='fro') ** 2)
                + beta * (norm(A, ord='fro') ** 2 + norm(B, ord='fro') ** 2)
                + r1 * np.trace(np.dot(U.T, np.dot(A.T, np.dot(lap_mat(drugMat), np.dot(A, U)))))
                + r2 * np.trace(np.dot(V.T, np.dot(B, np.dot(lap_mat(targetMat), np.dot(B.T, V))))))
        return loss

cdef class adam_opt:
    cdef float alpha, beta1, beta2, epsilon, t
    cdef np.ndarray m0, v0
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