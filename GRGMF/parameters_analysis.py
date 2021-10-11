# -*- coding: utf-8 -*-
# @author: Zichao Zhang
# @date:  Dec 2018
'''
this file is only used for test the performance of norm2 for A and B
@Alfred 2018/12/17
'''

import pandas as pd
from models.adpgmf import ADPGMF as _ADPGMF
from models.adpgmf import ADPGMF_gpu as _ADPGMF_gpu
import time
from functions import *
import matplotlib.pyplot as plt


def generate_data(dataset, cv_data, X, D, T, cvs, gpu=None):
    path = os.path.join(r'./result/final',
                        'result_' + dataset + '_' + str(cvs) + '.csv')
    if os.path.exists(path):
        result = pd.read_csv(path)
        result = result.sort_values(by='auc', ascending=False, kind='mergesort')
        best_dim = result.iloc[0, :]['mf_dim']
        best_resample = result.iloc[0, :]['resample']
        best_max_iter = result.iloc[0, :]['max_iter']
        best_beta = result.iloc[0, :]['beta']
        best_lamb = result.iloc[0, :]['lamb']
        best_r1 = result.iloc[0, :]['r1']
        best_r2 = result.iloc[0, :]['r2']
        result = result[result['mf_dim'] == best_dim]
        result = result[result['resample'] == best_resample]
        result = result[result['max_iter'] == best_max_iter]
        result = result.drop_duplicates(subset=['lamb', 'beta', 'r1', 'r2'])
        path = os.path.join(r'./result/final',
                            '_result_' + dataset + '_' + str(cvs) + '.csv')
        if not os.path.exists(path):
            h = open(path, 'a')
            h.write('lamb,beta,r1,r2,mf_dim,resample,auc,aupr,auc_conf,aupr_conf,time,h_mean,max_iter\n')
            line1 = result.iloc[0, :].values
            h.write('%.4f,%.4f,%.4f,%.4f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.6f,%d\n' % tuple(line1))
            h.close()
        else:
            line1 = result.iloc[[0]]
            data = pd.read_csv(path)
            data = line1.append(data, ignore_index=True)
            data = data.drop_duplicates(subset=['lamb', 'beta', 'r1', 'r2', 'mf_dim', 'resample', 'max_iter'])
            data.to_csv(path_or_buf=path, index=False)
        data = pd.read_csv(path)
    else:
        # currently we don't have ic3's data
        return
    pars = []
    dim = np.array([best_dim])
    resample = np.array([best_resample])
    max_iter = best_max_iter

    # r1 r2
    lamb = np.array([best_lamb])
    beta = np.array([best_beta])
    r1 = np.array([2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3]) * 0.5
    # #TODO: temperately use
    # if (cvs == 1) & (dataset == 'mndr'):
    #     r1 = np.array([0, 1.5 ** -5, 1.5 ** -4, 1.5 ** -3, 1.5 ** -2, 1.5 ** -1, 1.5 ** 0, 1.5 ** 1, 1.5 ** 2]) * 0.5
    r2 = np.array([2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3]) * 0.5
    # if (cvs == 1) & (dataset == 'hmdd'):
    #     r2 = np.array([1.5 ** -2, 1.5 ** -1, 1.5 ** 0, 1.5 ** 1, 1.5 ** 2, 1.5 ** 3, 1.5**4]) * 0.5
    pars += [[i, j, k, l, n, re] for i in lamb for j in beta for k in r1 for l in r2 for n in dim for re in resample]

    # #lamb
    # lamb = np.array([3. ** -2, 3. ** -1.5, 3. ** -1,3. ** -0.5, 3. ** 0, 3. ** 0.5, 3. ** 1, 3. ** 1.5, 3. ** 2,
    #                  3. ** 2.5]) * 0.3

    lamb = np.array([3. ** -3, 3. ** -2.5, 3. ** -2, 3. ** -1.5, 3. ** -1, 3. ** -0.5, 3. ** 0, 3. ** 0.5, 3. ** 1,
                     3. ** 1.5, 3. ** 2, 3. ** 2.5, 3. ** 3, 3. ** 3.5, 3. ** 4]) * 0.1

    beta = np.array([best_beta])
    r1 = np.array([best_r1])
    r2 = np.array([best_r2])
    pars += [[i, j, k, l, n, re] for i in lamb for j in beta for k in r1 for l in r2 for n in dim for re in resample]

    # beta
    lamb = np.array([best_lamb])
    # beta = np.array([3. ** -3, 3. ** -2.5, 3. ** -2, 3. ** -1.5, 3. ** -1, 3. ** -0.5, 3. ** 0, 3. ** 0.5, 3. ** 1,
    #                  3. ** 1.5, 3. ** 2, 3. ** 2.5, 3. ** 3,  3. ** 3.5]) * 1.

    beta = np.array([2. ** -3, 2. ** -2, 2. ** -1, 2. ** 0, 2. ** 1, 2. ** 2, 2. ** 3, 2. ** 4]) * 1.
    r1 = np.array([best_r1])
    r2 = np.array([best_r2])
    pars += [[i, j, k, l, n, re] for i in lamb for j in beta for k in r1 for l in r2 for n in dim for re in resample]

    args = {'K': 5, 'max_iter': max_iter, 'lr': 0.1, 'lamb': 0.3, 'beta': 0.1, 'r1': 0.3, 'r2': 0.6,
            'mf_dim': 50, 'c': 5, 'pre_end': False, 'cvs': cvs}

    if gpu == None:
        if (X.shape[0] <= 300) and (X.shape[1] <= 300):
            ADPGMF = _ADPGMF
        else:
            ADPGMF = _ADPGMF_gpu
    elif gpu == True:
        ADPGMF = _ADPGMF_gpu
    else:
        ADPGMF = _ADPGMF
    for par in pars:
        if len(par) == 5:
            i, j, k, l, n = par
            re = 1
        else:
            i, j, k, l, n, re = par
        if not data[(data['lamb'] == round(i, 4)) & (data['beta'] == round(j, 4)) & (data['r1'] == round(k, 4))
                    & (data['r2'] == round(l, 4))].empty:
            continue
        args['lamb'] = i
        args['beta'] = j
        args['r1'] = k
        args['r2'] = l
        args['mf_dim'] = int(n)
        args['resample'] = int(re)
        tic = time.clock()
        model = ADPGMF(max_iter=args['max_iter'], c=args['c'], resample=args['resample'], lamb=args['lamb'],
                       beta=args['beta'], r1=args['r1'], r2=args['r2'], lr=args['lr'], mf_dim=args['mf_dim'],
                       pre_end=args['pre_end'], K=args['K'], cvs=args['cvs'], verpose=2 ** 15)
        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        tatime = time.clock() - tic
        print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.2f\n" % (
            auc_avg, aupr_avg, auc_conf, aupr_conf, tatime)))
        h = open(path, 'a')
        h.write('%.4f,%.4f,%.4f,%.4f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.6f,%d\n' % (
            i, j, k, l, n, re, auc_avg, aupr_avg, auc_conf, aupr_conf, tatime, 2. / (1. / auc_avg + 1. / aupr_avg),
            max_iter))
        h.close()


def draw_curve(dataset, cvs):
    fontsize = 14
    font = {'size': fontsize}
    # drawing
    path = os.path.join(r'./result/final', '_result_' + dataset + '_' + str(cvs) + '.csv')
    data = pd.read_csv(path)

    best_dim = data.iloc[0, :]['mf_dim']
    best_resample = data.iloc[0, :]['resample']
    best_max_iter = data.iloc[0, :]['max_iter']
    best_beta = data.iloc[0, :]['beta']
    best_lamb = data.iloc[0, :]['lamb']
    best_r1 = data.iloc[0, :]['r1']
    best_r2 = data.iloc[0, :]['r2']
    # data = data.iloc[1:,:]
    data = data.drop_duplicates(subset=['lamb', 'beta', 'r1', 'r2', 'max_iter', 'resample', 'mf_dim'])

    # lambda
    draw_data = data[(data['beta'] == round(best_beta, 4))
                     & (data['mf_dim'] == round(best_dim, 4))
                     & (data['resample'] == round(best_resample, 4))
                     & (data['max_iter'] == round(best_max_iter, 4))
                     & (data['r1'] == round(best_r1, 4))
                     & (data['r2'] == round(best_r2, 4))]
    draw_data = draw_data.sort_values(by='lamb')
    x = draw_data['lamb'].values
    y = draw_data['auc'].values
    plt.figure()
    plt.plot(np.arange(len(y)), y)
    xtag = [r"$3^{-3}$", r"$3^{-2}$", r"$3^{-1}$", r"$3^{-0}$", r"$3^{1}$", r"$3^{2}$", r"$3^{3}$", ]
    plt.tick_params(labelsize=fontsize)
    plt.xticks(np.arange(0, len(x), 2), xtag)
    plt.xlabel(r'$\lambda$ (x0.1)', font)
    plt.ylabel('AUC', font)
    # plt.title('Dataset: ' + dataset + '    CVS: ' + str(cvs))
    plt.show()

    # beta
    draw_data = data[(data['lamb'] == round(best_lamb, 4))
                     & (data['mf_dim'] == round(best_dim, 4))
                     & (data['resample'] == round(best_resample, 4))
                     & (data['max_iter'] == round(best_max_iter, 4))
                     & (data['r1'] == round(best_r1, 4))
                     & (data['r2'] == round(best_r2, 4))]
    draw_data = draw_data.sort_values(by='beta')
    x = draw_data['beta'].values
    y = draw_data['auc'].values
    plt.figure()
    plt.plot(np.arange(len(y)), y)
    xtag = [r"$2^{-3}$", r"$2^{-2}$", r"$2^{-1}$", r"$2^{0}$", r"$2^{1}$", r"$2^{2}$", r"$2^{3}$", r"$2^{4}$"]
    plt.tick_params(labelsize=fontsize)
    plt.xticks(np.arange(0, len(x), 1), xtag)
    plt.xlabel(r'$\beta$', font)
    plt.ylabel('AUC', font)
    # plt.title('Dataset: ' + dataset + '    CVS: ' + str(cvs))
    plt.show()

    # r1 & r2
    draw_data = data[(data['lamb'] == round(best_lamb, 4))
                     & (data['beta'] == round(best_beta, 4))
                     & (data['mf_dim'] == round(best_dim, 4))
                     & (data['resample'] == round(best_resample, 4))
                     & (data['max_iter'] == round(best_max_iter, 4))]
    draw_data = draw_data.sort_values(by=['r1', 'r2'])
    X = draw_data['r1'].drop_duplicates().values
    Y = draw_data['r2'].drop_duplicates().values
    Z = draw_data['auc'].values.reshape(len(X), len(Y))
    # in order to aligment to meshgrid order
    Z = Z.T
    xtag = [r"$2^{-4}$", r"$2^{-3}$", r"$2^{-2}$", r"$2^{-1}$", r"$2^{0}$", r"$2^{1}$", r"$2^{2}$"]
    ytag = [r"$2^{-4}$", r"$2^{-3}$", r"$2^{-2}$", r"$2^{-1}$", r"$2^{0}$", r"$2^{1}$", r"$2^{2}$"]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x, y = np.meshgrid(np.arange(len(X)), np.arange(len(Y)))

    # ax.set_xlabel('r1', font)
    # ax.set_ylabel('r2', font)
    # ax.set_zlabel('AUC', font)
    plt.tick_params(labelsize=fontsize)
    ax.set_xticks(np.arange(len(X)))
    ax.set_xticklabels(xtag)
    ax.set_yticks(np.arange(len(Y)))
    ax.set_yticklabels(ytag)

    if dataset == 'ic':
        surf = ax.plot_wireframe(x, y, Z, color='b')
        ax.set_zlim(0.965, 0.99)
    else:
        surf = ax.plot_wireframe(x, y, Z, color='g')
        ax.set_zlim(0.9, 0.94)

    # ax.set_title('Dataset: ' + dataset + '    CVS: ' + str(cvs))
    plt.show()


def draw_curve_together(datasets: list, cvs):
    '''
    Note: No more than two curve.
    '''
    fontsize = 14
    font = {'size': fontsize}
    figure_list = [None, None]
    lamb_lns = []
    beta_lns = []
    colormap = ['ok-', '*g-', 'xb-']

    for dataset in datasets:
        # drawing
        col = colormap.pop()
        path = os.path.join(r'./result/final', '_result_' + dataset + '_' + str(cvs) + '.csv')
        data = pd.read_csv(path)
        best_dim = data.iloc[0, :]['mf_dim']
        best_resample = data.iloc[0, :]['resample']
        best_max_iter = data.iloc[0, :]['max_iter']
        best_beta = data.iloc[0, :]['beta']
        best_lamb = data.iloc[0, :]['lamb']
        best_r1 = data.iloc[0, :]['r1']
        best_r2 = data.iloc[0, :]['r2']
        # data = data.iloc[1:,:]
        data = data.drop_duplicates(subset=['lamb', 'beta', 'r1', 'r2', 'max_iter', 'resample', 'mf_dim'])

        # lambda
        draw_data = data[(data['beta'] == round(best_beta, 4))
                         & (data['mf_dim'] == round(best_dim, 4))
                         & (data['resample'] == round(best_resample, 4))
                         & (data['max_iter'] == round(best_max_iter, 4))
                         & (data['r1'] == round(best_r1, 4))
                         & (data['r2'] == round(best_r2, 4))]
        draw_data = draw_data.sort_values(by='lamb')
        x = draw_data['lamb'].values
        y = draw_data['auc'].values

        if figure_list[0]:
            f_lamb = plt.figure(figure_list[0])
            lamb_ax = lamb_ax.twinx()
        else:
            f_lamb = plt.figure()
            figure_list[0] = f_lamb.number
            lamb_ax = f_lamb.add_subplot(111)
            xtag = [r"$3^{-3}$", r"$3^{-2}$", r"$3^{-1}$", r"$3^{-0}$", r"$3^{1}$", r"$3^{2}$", r"$3^{3}$", ]
            lamb_ax.set_xlabel(r'$\lambda$ (x0.1)', font)
            plt.tick_params(labelsize=fontsize)
            lamb_ax.set_xticks(np.arange(0, len(x), 2))
            lamb_ax.set_xticklabels(xtag)
        plt.tick_params(labelsize=fontsize)
        lamb_ax.set_ylabel('AUC(' + dataset.upper() + ')', font)
        lamb_lns += lamb_ax.plot(np.arange(len(y)), y, col)

        # beta
        draw_data = data[(data['lamb'] == round(best_lamb, 4))
                         & (data['mf_dim'] == round(best_dim, 4))
                         & (data['resample'] == round(best_resample, 4))
                         & (data['max_iter'] == round(best_max_iter, 4))
                         & (data['r1'] == round(best_r1, 4))
                         & (data['r2'] == round(best_r2, 4))]
        draw_data = draw_data.sort_values(by='beta')
        x = draw_data['beta'].values
        y = draw_data['auc'].values

        if figure_list[1]:
            f_beta = plt.figure(figure_list[1])
            beta_ax = beta_ax.twinx()
        else:
            f_beta = plt.figure()
            figure_list[1] = f_beta.number
            beta_ax = f_beta.add_subplot(111)
            xtag = [r"$2^{-3}$", r"$2^{-2}$", r"$2^{-1}$", r"$2^{0}$", r"$2^{1}$", r"$2^{2}$", r"$2^{3}$", r"$2^{4}$"]
            beta_ax.set_xlabel(r'$\beta$', font)
            plt.tick_params(labelsize=fontsize)
            beta_ax.set_xticks(np.arange(0, len(x), 1))
            beta_ax.set_xticklabels(xtag)
        plt.tick_params(labelsize=fontsize)
        beta_ax.set_ylabel('AUC(' + dataset.upper() + ')', font)
        beta_lns += beta_ax.plot(np.arange(len(y)), y, col)

    plt.figure(figure_list[0])
    datasets_ = [i.upper() for i in datasets]
    lamb_ax.legend(lamb_lns, datasets_, loc=0)
    plt.show()

    plt.figure(figure_list[1])
    beta_ax.legend(beta_lns, datasets_, loc=0)
    plt.show()


def draw_curve_together2(datasets: list, cvs):
    '''
    Note: No more than two curve.
    '''
    fontsize = 14
    font = {'size': fontsize}
    figure_list = [None, None]
    lamb_lns = []
    beta_lns = []
    colormap = ['ok-', '*g-', 'xb-']

    for dataset in datasets:
        # drawing
        col = colormap.pop()
        path = os.path.join(r'./result/final', '_result_' + dataset + '_' + str(cvs) + '.csv')
        data = pd.read_csv(path)
        best_dim = data.iloc[0, :]['mf_dim']
        best_resample = data.iloc[0, :]['resample']
        best_max_iter = data.iloc[0, :]['max_iter']
        best_beta = data.iloc[0, :]['beta']
        best_lamb = data.iloc[0, :]['lamb']
        best_r1 = data.iloc[0, :]['r1']
        best_r2 = data.iloc[0, :]['r2']
        # data = data.iloc[1:,:]
        data = data.drop_duplicates(subset=['lamb', 'beta', 'r1', 'r2', 'max_iter', 'resample', 'mf_dim'])

        # lambda
        draw_data = data[(data['beta'] == round(best_beta, 4))
                         & (data['mf_dim'] == round(best_dim, 4))
                         & (data['resample'] == round(best_resample, 4))
                         & (data['max_iter'] == round(best_max_iter, 4))
                         & (data['r1'] == round(best_r1, 4))
                         & (data['r2'] == round(best_r2, 4))]
        draw_data = draw_data.sort_values(by='lamb')
        x = draw_data['lamb'].values
        y = draw_data['auc'].values
        temp = np.array([0.1 * 3. ** -3, 0.1 * 3. ** -2, 0.1 * 3. ** -1, 0.1 * 3. ** 0, 0.1 * 3. ** 1, 0.1 * 3. ** 2,
                         0.1 * 3. ** 3, 0.1 * 3. ** 4, ])
        temp = np.round(temp, 3)
        temp_ind = [np.round(i, 3) in temp for i in x]
        x = x[temp_ind]
        y = y[temp_ind]

        if figure_list[0]:
            f_lamb = plt.figure(figure_list[0])
            # lamb_ax = lamb_ax.twinx()
        else:
            f_lamb = plt.figure()
            figure_list[0] = f_lamb.number
            lamb_ax = f_lamb.add_subplot(111)
            xtag = [r"$3^{-3}$", r"$3^{-2}$", r"$3^{-1}$", r"$3^{0}$", r"$3^{1}$", r"$3^{2}$", r"$3^{3}$", r"$3^{4}$"]

            lamb_ax.set_xlabel(r'$\lambda$ (x0.1)', font)
            plt.tick_params(labelsize=fontsize)
            lamb_ax.set_xticks(np.arange(0, len(x), 1))
            lamb_ax.set_xticklabels(xtag)
            lamb_ax.set_ylim(0.9, 1)
            lamb_ax.set_ylabel('AUC', font)
        lamb_lns += lamb_ax.plot(np.arange(len(y)), y, col)

        # beta
        draw_data = data[(data['lamb'] == round(best_lamb, 4))
                         & (data['mf_dim'] == round(best_dim, 4))
                         & (data['resample'] == round(best_resample, 4))
                         & (data['max_iter'] == round(best_max_iter, 4))
                         & (data['r1'] == round(best_r1, 4))
                         & (data['r2'] == round(best_r2, 4))]
        draw_data = draw_data.sort_values(by='beta')
        x = draw_data['beta'].values
        y = draw_data['auc'].values
        temp = np.array([2. ** -3, 2. ** -2, 2. ** -1, 2. ** 0, 2. ** 1, 2. ** 2, 2. ** 3, 2. ** 4])
        temp = np.round(temp, 3)
        temp_ind = [np.round(i, 3) in temp for i in x]
        x = x[temp_ind]
        y = y[temp_ind]

        if figure_list[1]:
            f_beta = plt.figure(figure_list[1])
            # beta_ax = beta_ax.twinx()
        else:
            f_beta = plt.figure()
            figure_list[1] = f_beta.number
            beta_ax = f_beta.add_subplot(111)
            xtag = [r"$2^{-3}$", r"$2^{-2}$", r"$2^{-1}$", r"$2^{0}$", r"$2^{1}$", r"$2^{2}$", r"$2^{3}$", r"$2^{4}$"]
            beta_ax.set_xlabel(r'$\beta$', font)
            plt.tick_params(labelsize=fontsize)
            beta_ax.set_xticks(np.arange(0, len(x), 1))
            beta_ax.set_xticklabels(xtag)
            beta_ax.set_ylim(0.9, 1)
            beta_ax.set_ylabel('AUC', font)
        beta_lns += beta_ax.plot(np.arange(len(y)), y, col)

    datasets_ = [i.upper() for i in datasets]
    plt.figure(figure_list[0])
    lamb_ax.legend(lamb_lns, datasets_, loc=4)
    plt.show()

    plt.figure(figure_list[1])
    beta_ax.legend(beta_lns, datasets_, loc=4)
    plt.show()


if __name__ == "__main__":
    datasets = ['nr', 'gpcr', 'ic', 'e', 'mndr', 'hmdd']
    cvses = [1, 2, 3]
    seeds = [7771, 8367, 22, 1812, 4659]
    data_dir = os.path.join(os.path.pardir, 'data')
    #
    # datasets = ['ic', 'hmdd']
    # cvses = [1]
    # for dataset in datasets:
    #     for cvs in cvses:
    #         intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))
    #         if cvs == 1:  # CV setting CVS1
    #             X, D, T, cv = intMat, drugMat, targetMat, 1
    #         if cvs == 2:  # CV setting CVS2
    #             X, D, T, cv = intMat, drugMat, targetMat, 0
    #         if cvs == 3:  # CV setting CVS3
    #             X, D, T, cv = intMat.T, targetMat, drugMat, 0
    #         cv_data = cross_validation(X, seeds, cv)
    #         generate_data(dataset, cv_data, X, D, T, cvs)

    datasets = ['ic', 'hmdd']
    # cvses = [1,]
    # for dataset in datasets:
    #     for cvs in cvses:
    #         if os.path.exists(os.path.join(r'./result/final','result_' + dataset + '_' + str(cvs) + '.csv')):
    #             draw_curve(dataset, cvs)
    draw_curve_together2(datasets, cvs=1)
