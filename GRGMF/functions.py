import os
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import rbf_kernel



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data_from_file(dataset, folder):
    def load_rating_file_as_matrix(filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        data = []
        stat = []
        try:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = np.array(line.split("\t"), dtype='int')
                    stat.append(sum(arr))
                    data.append(arr)
                    line = f.readline()
        except:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.split("\t")
                    try:
                        arr = arr[1:]
                        arr = np.array(arr, dtype='int')
                        stat.append(sum(arr))
                        data.append(arr)
                    except:
                        pass
                    line = f.readline()

        # Construct matrix
        mat = np.array(data)
        return mat

    def load_matrix(filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        data = []
        # for the situation that data files contain col/row name
        try:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = np.array(line.split("\t"), dtype='float')
                    data.append(arr)
                    line = f.readline()
        except:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.split("\t")
                    try:
                        arr = arr[1:]
                        arr = np.array(arr, dtype='float')
                        data.append(arr)
                    except:
                        pass
                    line = f.readline()
        mat = np.array(data)
        return mat
    '''
    int_array = load_rating_file_as_matrix(os.path.join(folder, dataset + "_admat_dgc.txt"))

    drug_sim = load_matrix(os.path.join(folder, dataset + "_simmat_dc.txt"))
    target_sim = load_matrix(os.path.join(folder, dataset + "_simmat_dg.txt"))
    '''
    drugMat = np.loadtxt(os.path.join(folder, dataset, "DrugSimMat1"),dtype='float64')
    intMat = np.loadtxt(os.path.join(folder, dataset,"DrugDisease"),delimiter=',',dtype='float64')
    targetMat = np.loadtxt(os.path.join(folder, dataset,"DiseaseSimMat"),delimiter=' ',dtype='float64')
    '''
    intMat = np.array(int_array, dtype=np.float64).T  # drug-target interaction matrix
    drugMat = np.array(drug_sim, dtype=np.float64)  # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    '''

    return intMat, drugMat, targetMat


def get_drugs_targets_names(dataset, folder):
    with open(os.path.join(folder, dataset + "/Drug-Target Interactions"), "r") as inf:
        drugs = next(inf).strip("\n").split('\t')
        targets = [line.strip("\n").split('\t')[0] for line in inf]
        if '' in drugs:
            drugs.remove('')
        if '' in targets:
            targets.remove('')
    return drugs, targets


def cross_validation(intMat, seeds, cv=0, num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = index.size // num
        for i in range(num):
            if i < num - 1:
                ii = index[i * step:(i + 1) * step]
            else:
                ii = index[i * step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k / num_targets, k % num_targets] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data


def train(model, cv_data, intMat, drugMat, targetMat):
    aupr, auc = [], []
    for seed in list(cv_data.keys()):
        for W, test_data, test_label in cv_data[seed]:
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val = model.evaluation(test_data, test_label)
            aupr.append(aupr_val)
            auc.append(auc_val)
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64)


def svd_init(M, num_factors):
    from scipy.linalg import svd
    U, s, V = svd(M, full_matrices=False)
    ii = np.argsort(s)[::-1][:num_factors]
    s1 = np.sqrt(np.diag(s[ii]))
    U0, V0 = U[:, ii].dot(s1), s1.dot(V[ii, :])
    return U0, V0.T


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)

def WKNKN(Y, SD, ST, K, ita):                 #add
    Yd = np.zeros(Y.shape)
    Yt = np.zeros(Y.shape)
    wi = np.zeros((K,))
    wj = np.zeros((K,))
    num_drugs, num_targets = Y.shape
    for i in np.arange(num_drugs):
        dnn_i = np.argsort(SD[i,:])[::-1][1:K+1]
        Zd = np.sum(SD[i, dnn_i])
        for ii in np.arange(K):
            wi[ii] = (ita ** (ii)) * SD[i,dnn_i[ii]]
        if not np.isclose(Zd, 0.):
            Yd[i,:] = np.sum(np.multiply(wi.reshape((K,1)), Y[dnn_i,:]), axis=0) / Zd
    for j in np.arange(num_targets):
        tnn_j = np.argsort(ST[j, :])[::-1][1:K+1]
        Zt = np.sum(ST[j, tnn_j])
        for jj in np.arange(K):
            wj[jj] = (ita ** (jj)) * ST[j,tnn_j[jj]]
        if not np.isclose(Zt, 0.):
            Yt[:,j] = np.sum(np.multiply(wj.reshape((1,K)), Y[:,tnn_j]), axis=1) / Zt
    Ydt = (Yd + Yt)/2
    x, y = np.where(Ydt > Y)

    Y_tem = Y.copy()
    Y_tem[x, y] = Ydt[x, y]
    return Y_tem