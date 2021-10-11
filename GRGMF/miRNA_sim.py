# -*- coding: utf-8 -*-
# @author: Zichao Zhang
# @date:  Jan 2018

import numpy as np
import pandas as pd
import os
from scipy import sparse
import multiprocessing

_mR_list = None
_mti = None
_netm = None

def data_collection():
    # this part if for data collection
    mR_list = pd.read_csv(os.path.join('../data', 'hmdd2.0_miRNA_name.csv'), header=None, names=['miRNA name'])
    humanNet = pd.read_csv(os.path.join('../data', 'HumanNet.v1.csv'))
    MTI = pd.read_csv(os.path.join('../data', 'hsa_MTI.csv'))
    mod_mti = pd.DataFrame(columns=MTI.columns)
    mod_net = pd.DataFrame(columns=humanNet.columns)
    G_all = set()
    for i in range(len(mR_list)):
        mR_i = mR_list.iloc[i].values[0].replace('mir', 'miR')
        print('miRNA: %s' % mR_i)
        MTI_i = MTI[MTI['miRNA'].str.contains('^' + mR_i)]
        mod_mti = mod_mti.append(MTI_i, ignore_index=True)
        mod_mti = mod_mti.drop_duplicates(subset = ['miRNA', 'Target Gene (Entrez Gene ID)'])
        G_all = G_all | set(mod_mti['Target Gene (Entrez Gene ID)'].values)

    for g in G_all:
        print('gene ID: %s' % g)
        mod_net = mod_net.append(humanNet[(humanNet['gene_i'] == g)],ignore_index=True)
        mod_net = mod_net.append(humanNet[(humanNet['gene_j'] == g)], ignore_index=True)
    mod_mti = mod_mti.drop_duplicates(subset=['miRNA', 'Target Gene (Entrez Gene ID)'])
    mod_net = mod_net.drop_duplicates(subset=['gene_i', 'gene_j'])
    mod_mti.to_csv(path_or_buf='../data/mod_mti.csv')
    mod_net.to_csv(path_or_buf='../data/mod_net.csv')
    pass

def single_sim(index):
    global _mR_list
    global _mti
    global _netm
    i, j = index
    print("i: %d, j: %d" % (i, j))
    if i == j:
        return 1.
    else:
        # gene set associated with i/j-th mirna
        mR_i = _mR_list.iloc[i].values[0].replace('mir', 'miR')
        mR_j = _mR_list.iloc[j].values[0].replace('mir', 'miR')
        Gi = np.array(
            list(set(_mti[_mti['miRNA'].str.contains('^' + mR_i)]['Target Gene (Entrez Gene ID)'].values)))
        Gj = np.array(
            list(set(_mti[_mti['miRNA'].str.contains('^' + mR_j)]['Target Gene (Entrez Gene ID)'].values)))
        # miRNA may be associated with some gene existed in MTI dataset but not in humanNetV1.0
        Gi = Gi[Gi <= _netm.shape[0]]
        Gj = Gj[Gj <= _netm.shape[0]]
        if (len(Gi) == 0) or (len(Gj) == 0):
            return 0.
        else:
            g2g = np.zeros([len(Gi), len(Gj)])
            idx2 = np.array([[n, k] for n in range(len(Gi)) for k in range(len(Gj))])
            idx2 = idx2.T
            g2g[idx2[0], idx2[1]] = _netm[Gi[idx2[0]], Gj[idx2[1]]].toarray()
            return (g2g.max(axis=1).sum() + g2g.max(axis=0).sum()) / (len(Gi) + len(Gj))

def miRNA_sim(num_procss):
    global _mR_list
    global _mti
    global _netm
    dataset = 'hmdd'
    hnet = pd.read_csv(os.path.join('../data', 'mod_net.csv'))
    mti = pd.read_csv(os.path.join('../data', 'mod_mti.csv'))
    mR_list = pd.read_csv(os.path.join('../data', 'hmdd2.0_miRNA_name.csv'), header=None, names=['miRNA name'])
    mR_sim = np.zeros([len(mR_list), len(mR_list)])
    idx = np.triu_indices_from(mR_sim)
    netm = sparse.dok_matrix((max([hnet['gene_i'].max() + 1, hnet['gene_j'].max() + 1]),
                              max([hnet['gene_i'].max() + 1, hnet['gene_j'].max() + 1])),
                             dtype=np.double)
    # symmetric matrix
    netm[hnet['gene_i'].values, hnet['gene_j'].values] = hnet['LLS_N'].values
    netm[hnet['gene_j'].values, hnet['gene_i'].values] = hnet['LLS_N'].values
    # diag shoube 1.
    netm.setdiag(1.)
    _mR_list = mR_list
    _mti = mti
    _netm = netm
    if num_procss >1:
        pool = multiprocessing.Pool(processes=num_procss)
        res = np.array(pool.map(single_sim, list(zip(idx[0], idx[1]))))
        res.flatten()
        pool.close()
        pool.join()
        mR_sim[idx[0], idx[1]] = res

    else:
        for i, j in zip(idx[0], idx[1]):
            print("i: %d, j: %d" % (i, j))
            if i == j:
                mR_sim[i, j] = 1.
            else:
                # gene set associated with i/j-th mirna
                mR_i = mR_list.iloc[i].values[0].replace('mir', 'miR')
                mR_j = mR_list.iloc[j].values[0].replace('mir', 'miR')
                Gi = np.array(list(set(mti[mti['miRNA'].str.contains('^' + mR_i)]['Target Gene (Entrez Gene ID)'].values)))
                Gj = np.array(list(set(mti[mti['miRNA'].str.contains('^' + mR_j)]['Target Gene (Entrez Gene ID)'].values)))
                # miRNA may be associated with some gene existed in MTI dataset but not in humanNetV1.0
                Gi = Gi[Gi <= netm.shape[0]]
                Gj = Gj[Gj <= netm.shape[0]]
                if (len(Gi) == 0) or (len(Gj) == 0):
                    mR_sim[i, j] = 0.
                else:
                    g2g = np.zeros([len(Gi), len(Gj)])
                    idx2 = np.array([[n, k] for n in range(len(Gi)) for k in range(len(Gj))])
                    idx2 = idx2.T
                    g2g[idx2[0], idx2[1]] = netm[Gi[idx2[0]], Gj[idx2[1]]].toarray()
                    mR_sim[i, j] = (g2g.max(axis=1).sum() + g2g.max(axis=0).sum()) / (len(Gi) + len(Gj))
    # similarity matrix is a symmetric matrix
    mR_sim[idx[1], idx[0]] = mR_sim[idx[0], idx[1]]
    np.savetxt(os.path.join('../data/datasets', dataset + "_simmat_dg.txt"), mR_sim, fmt='%.9f', delimiter='\t')


if __name__ == "__main__":
    # don't use multiprocess in windows
    miRNA_sim(num_procss=1)