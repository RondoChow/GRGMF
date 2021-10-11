import os
import datetime
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc as AUC
from time import time
# for accelarating the computation
# the function fix have been implemented with cython
#from cython_func import pbmda_fix

class PBMDA():
    def __init__(self, Max_length=3, Pa=2.26, Similarity_threshold=0.5):
        self.Max_length = Max_length
        self.Pa = Pa
        self.Similarity_threshold = Similarity_threshold
        self.n = 0

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None, ):
        self.n += 1
        t = time()
        self.P = np.zeros(intMat.shape) # P here is transposed comparing to the origin P defined in origin pbmda code
        self.P = pbmda_fix(intMat * W, drugMat, targetMat, self.Similarity_threshold, self.Pa, self.Max_length)
        self.t = time() - t

    def test(self, test_data):
        test_data = test_data.T
        y_pre = self.P[test_data[0], test_data[1]]
        return y_pre

    def evaluation(self, test_data, test_label, verpose=1):
        test_data = test_data.T
        y_pre = self.P[test_data[0], test_data[1]]
        fpr, tpr, thresholds = roc_curve(test_label, y_pre)
        p, r, thresholds = precision_recall_curve(test_label, y_pre)
        auc_val = AUC(fpr, tpr)
        aupr_val = AUC(r, p)
        print("Data[%d]: AUC: %.4f, AUPR: %.4f, time: %.3f" % (self.n, auc_val, aupr_val, self.t))
        return aupr_val, auc_val

    def __str__(self):
        return "Model: PBMDA, Max_length: %s, Pa: %s, Similarity_threshold: %s" % (self.Max_length, self.Pa,
                                                                                   self.Similarity_threshold)

def pbmda_fix(intMat, drugMat, targetMat,Similarity_threshold, Pa, Max_length):
    A = drugMat.copy()
    B = targetMat.copy()
    Y = intMat.copy()
    y, x = Y.shape  # y: num of miRNAs, x: num of diseases
    P = np.zeros([x, y])
    print("Step 5. constructed a heterogeneous graph consisting of three interlinked sub-graphs..")
    # constructed a heterogeneous graph consisting of three interlinked sub-graphs
    Network_disease = np.zeros([x, 4, max(x, y)])  # 0:disease 1:miRNA
    Network_miRNA = np.zeros([y, 4, max(x, y)])

    for i in range(x):
        index = 0
        for j in range(x):
            if B[i][j] > Similarity_threshold:
                Network_disease[i][0][index] = B[i][j]  # disease-disease  similarity
                Network_disease[i][1][index] = int(j + 1)
                index += 1
        index = 0
        for k in range(y):
            if Y[k][i] == 1:
                Network_disease[i][2][index] = 1  # disease-miRNA similarity
                Network_disease[i][3][index] = int(k + 1)
                index += 1

    for i in range(y):
        index = 0
        for j in range(y):
            if A[i][j] > Similarity_threshold:
                Network_miRNA[i][2][index] = A[i][j]  # miRNA-miRNA  similarity
                Network_miRNA[i][3][index] = int(j + 1)
                index += 1
        index = 0
        for k in range(x):
            if Y[i][k] == 1:
                Network_miRNA[i][0][index] = 1  # disease-disease similarity
                Network_miRNA[i][1][index] = int(k + 1)
                index += 1
                ####################################################################

    ####################################################################
    print("Step 6. start the depth-first search algorithm..")
    # It is a little complicated to explain the process.
    # For example, this part works for the first iteration

    # for i in range(x):  #+:disease -:miRNA
    for i in range(x):  # +:disease -:miRNA
        # print(i)
        l = []
        ll = []
        lll = [i + 1]

        for k in range(max(y, x)):  # disease
            if Network_disease[i][0][k] == 0:
                break
            if k != i:
                lll.append(int(Network_disease[i][1][k]))
                ll.append(lll)
                lll = [i + 1]
        for k in range(max(y, x)):  # miRNA
            if Network_disease[i][2][k] == 0:
                break
            lll.append(-int(Network_disease[i][3][k]))
            ll.append(lll)
            lll = [i + 1]
            P[i][-int(ll[-1][-1]) - 1] += 1
        ####################################################################

        ####################################################################
        # This part worked for the rest iterations based on the selected maximun path length
        # for j in range(Max_length-1):
        for j in range(Max_length - 1):
            ll1 = []
            for k in range(len(ll)):
                # unit=ll[k]
                lll = ll[k]
                if ll[k][j + 1] > 0:  # disease
                    for m in range(max(y, x)):  # disease
                        if Network_disease[int(ll[k][j + 1] - 1)][0][m] == 0:
                            break;
                        if (Network_disease[int(ll[k][j + 1] - 1)][1][m]) not in ll[k]:
                            ll1.append(ll[k] + [Network_disease[int(ll[k][j + 1] - 1)][1][m]])
                    for m in range(max(y, x)):  # miRNA
                        temp = 1
                        if Network_disease[int(ll[k][j + 1] - 1)][2][m] == 0:
                            break;
                        if (-(Network_disease[int(ll[k][j + 1] - 1)][3][m])) not in ll[k]:
                            ll1.append(ll[k] + [-(Network_disease[int(ll[k][j + 1] - 1)][3][m])])
                            for q in range(j + 2):
                                if ll1[-1][q + 1] > 0 and ll1[-1][q] > 0:  # disease-disease
                                    temp *= B[int(ll1[-1][q] - 1)][int(ll1[-1][q + 1] - 1)]
                                elif ll1[-1][q + 1] < 0 and ll1[-1][q] > 0:  # disease-miRNA
                                    temp *= 1
                                elif ll1[-1][q + 1] > 0 and ll1[-1][q] < 0:  # miRNA-disease
                                    temp *= 1
                                elif ll1[-1][q + 1] < 0 and ll1[-1][q] < 0:  # miRNA-miRNA
                                    temp *= A[-int(ll1[-1][q]) - 1][-int(ll1[-1][q + 1]) - 1]
                            P[i][-int(ll1[-1][-1]) - 1] += (temp ** (Pa * (j + 2)))

                if ll[k][j + 1] < 0:  # miRNA
                    for m in range(max(y, x)):  # disease
                        if Network_miRNA[-int(ll[k][j + 1]) - 1][0][m] == 0:
                            break;
                        if Network_miRNA[-int(ll[k][j + 1]) - 1][1][m] not in ll[k]:
                            ll1.append(ll[k] + [Network_miRNA[-int(ll[k][j + 1]) - 1][1][m]])
                    for m in range(max(y, x)):  # miRNA
                        temp = 1
                        if Network_miRNA[-int(ll[k][j + 1]) - 1][2][m] == 0:
                            break;
                        if (-Network_miRNA[-int(ll[k][j + 1]) - 1][3][m]) not in ll[k]:
                            ll1.append(ll[k] + [-(Network_miRNA[-int(ll[k][j + 1]) - 1][3][m])])
                            for q in range(j + 2):
                                if ll1[-1][q + 1] > 0 and ll1[-1][q] > 0:  # disease-disease
                                    temp *= B[int(ll1[-1][q] - 1)][int(ll1[-1][q + 1] - 1)]
                                elif ll1[-1][q + 1] < 0 and ll1[-1][q] > 0:  # miRNA-disease
                                    temp *= 1
                                elif ll1[-1][q + 1] > 0 and ll1[-1][q] < 0:  # miRNA-disease
                                    temp *= 1
                                elif ll1[-1][q + 1] < 0 and ll1[-1][q] < 0:  # miRNA-miRNA
                                    temp *= A[-int(ll1[-1][q]) - 1][-int(ll1[-1][q + 1]) - 1]
                            # The end of each iterations need to be aggregated with the scores
                            P[i][-int(ll1[-1][-1]) - 1] += (temp ** (Pa * (j + 2)))

            ll = ll1
    return P.T
# This part is a little complated, I hope you can understand with the illustraion in the paper
###################################################################
