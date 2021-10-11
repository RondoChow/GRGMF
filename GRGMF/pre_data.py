import os
import numpy as np
from collections import defaultdict


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

for dataset in ['ld1', 'lnc2cancer', 'mndr']:
    folder = 'C:\\Users\Alfred\Desktop\data'
    drug_sim = load_matrix(os.path.join(folder, dataset + "_simmat_dc.txt"))
    target_sim = load_matrix(os.path.join(folder, dataset + "_simmat_dg.txt"))

    drug_sim1 = load_matrix(os.path.join(folder, dataset + "_simmat_dc1.txt"))
    target_sim1 = load_matrix(os.path.join(folder, dataset + "_simmat_dg1.txt"))

    drugMat = np.array(drug_sim, dtype=np.float64)  # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    drugMat1 = np.array(drug_sim1, dtype=np.float64)  # drug similarity matrix
    targetMat1 = np.array(target_sim1, dtype=np.float64)  # target similarity matrix

    Sd = (drugMat + drugMat1) / 2.
    St = (targetMat + targetMat1) / 2.
    np.savetxt(r"C:\Users\Alfred\GoogleDrive\Alfred's work\Project\data\datasets" + '\\' + dataset
               + "_simmat_dc.txt", Sd, fmt='%.9f', delimiter='\t')
    np.savetxt(r"C:\Users\Alfred\GoogleDrive\Alfred's work\Project\data\datasets" + '\\' + dataset
               + "_simmat_dg.txt", St, fmt='%.9f', delimiter='\t')

if __name__ == "__main__":
    folder = '../data/Data_mirna/'
    dataset = "hdmm_m"


    index = np.loadtxt(os.path.join(folder,  "knowndiseasemirnainteraction.txt"), dtype=int)
    index -= 1 #From 0
    index = tuple(index.T)
    Sd = np.loadtxt(os.path.join(folder,  "miRNA functional similarity.txt"))
    target_sim1 = np.loadtxt(os.path.join(folder, "disease semantic similarity 1.txt"))
    target_sim2 = np.loadtxt(os.path.join(folder, "disease semantic similarity 2.txt"))
    St = (target_sim1 + target_sim2) / 2.
    Intmat = np.zeros([len(Sd), len(St)])
    Intmat[index] = 1
    Intmat = Intmat.T

    tz = np.where(np.sum(St, 1)==1)[0]
    dz = np.where(np.sum(Sd, 1)==1)[0]

    St = np.delete(St, tz, 1)
    St = np.delete(St, tz, 0)
    Sd = np.delete(Sd, dz, 1)
    Sd = np.delete(Sd, dz, 0)
    Intmat = np.delete(Intmat, dz, 1)
    Intmat = np.delete(Intmat, tz, 0)


    np.savetxt(os.path.join('../data/datasets', dataset + "_admat_dgc.txt"), Intmat, fmt='%d', delimiter='\t')
    np.savetxt(os.path.join('../data/datasets', dataset + "_simmat_dc.txt"), Sd, fmt='%.9f', delimiter='\t')
    np.savetxt(os.path.join('../data/datasets', dataset + "_simmat_dg.txt"), St, fmt='%.9f', delimiter='\t')
