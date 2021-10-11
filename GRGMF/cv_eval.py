import time
from functions import *
from models import ADPGMF, NRLMF, NetLapRLS, BLMNII, WNNGIP, KBMF, CMF, MDHGI, CMFMDA, SIMCLDA, PBMDA, GRNMF, DRCC
import pickle
import pandas as pd


def adpgmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    # generate parameter combinations
    # int(np.random.rand() * 10000)
    np.random.seed(5110)
    lamb = np.arange(0, 1, 0.1) ** 2.
    beta = [0.] + np.power(2, np.arange(-5, 5, 2.)).tolist()
    r1 = np.power(2., np.arange(-3, 6, 1)).tolist()
    r2 = np.power(2., np.arange(-3, 6, 1)).tolist()
    dim = np.arange(25, 225, 25)
    nei = np.arange(0, 8, 1)
    emph = 2**np.arange(0, 6)
    resample = np.array([0])
    itas = np.array([0., 0.2])

    par = []
    for ii in range(4096):
        i = lamb[np.random.randint(0, len(lamb))]
        j = beta[np.random.randint(0, len(beta))]
        k = r1[np.random.randint(0, len(r1))]
        l = r2[np.random.randint(0, len(r2))]
        n = dim[np.random.randint(0, len(dim))]
        re = resample[np.random.randint(0, len(resample))]
        K = nei[np.random.randint(0, len(nei))]
        c = emph[np.random.randint(0, len(emph))]
        ita = itas[np.random.randint(0, len(itas))]
        par.append([i, j, k, l, n, re, K, c, ita])
    np.random.shuffle(par)

    num_par = len(par)
    # simplly no need to try them all.(also see Bergstra, James, and Yoshua Bengio.
    # "Random search for hyper-parameter optimization." Journal of Machine Learning Research 13.Feb (2012): 281-305.)
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists('./result/' + 'cvs' + str(cvs)):
        os.mkdir('./result/' + 'cvs' + str(cvs))
    f = 'result_' + str(dataset) + '_' + str(cvs) + time.strftime('_%y%m%d_%H%M.csv')
    pklfilename = 'var_' + dataset + '.pkl'
    # add header
    max_iter = 100
    args = {'K': 5, 'max_iter': max_iter, 'lr': 0.01, 'lamb': 0.3, 'beta': 0.1, 'r1': 0.3, 'r2': 0.6,
            'mf_dim': 50, 'c': 5, 'ita': 0.5, 'cvs': cvs}
    max_auc, auc_opt = 0, []

    # weather to  continue the former process
    if (para['cv_continue']) & (os.path.exists(os.path.join('./result/cvs' + str(cvs), pklfilename))):
        with open(os.path.join('./result/cvs' + str(cvs), pklfilename), 'rb') as pklhandler:
            par, max_auc, auc_opt, f = pickle.load(pklhandler)
    path = os.path.join('./result/cvs' + str(cvs), f)
    while len(par):
        if len(par[0]) == 5:
            i, j, k, l, n = par[0]
            re = 0
            K = 5
            c = 5
            ita = 0.5
        else:
            i, j, k, l, n, re, K, c, ita = par[0]
        args['lamb'] = i
        args['beta'] = j
        args['r1'] = k
        args['r2'] = l
        args['mf_dim'] = int(n)
        args['resample'] = int(re)
        args['K'] = int(K)
        args['c'] = int(c)
        args['ita'] = ita

        tic = time.clock()
        model = ADPGMF(max_iter=args['max_iter'], c=args['c'], resample=args['resample'], lamb=args['lamb'],
                       beta=args['beta'], r1=args['r1'], r2=args['r2'], lr=args['lr'], mf_dim=args['mf_dim'],
                       K=args['K'], cvs=args['cvs'], ita=args['ita'])
        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        tatime = time.clock() - tic
        print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.2f\n" % (
            auc_avg, aupr_avg, auc_conf, aupr_conf, tatime)))

        if not os.path.exists(path):
            h = open(path, 'a')
            h.write('lamb,beta,r1,r2,mf_dim,resample,K,c,ita,auc,aupr,auc_conf,aupr_conf,time,h_mean,max_iter\n')
            h.close()
        h = open(path, 'a')
        h.write('%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.2f,%.6f,%d\n' % (
            i, j, k, l, n, re, int(K), int(c), ita,auc_avg, aupr_avg, auc_conf, aupr_conf,
            tatime, 2. / (1. / auc_avg + 1. / aupr_avg), max_iter))
        h.close()
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
        del par[0]
        store_var = [par, max_auc, auc_opt, f]
        with open(os.path.join('./result/cvs' + str(cvs), pklfilename), 'wb') as pklhandler:
            pickle.dump(store_var, pklhandler)

        cmd = "Present optimal setting:\n%s\n" % auc_opt[0]
        cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Process: %d/%d \n" % (
            auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4], num_par - len(par), num_par)

        print(cmd)

    result = pd.read_csv(path)
    result = result.sort_values(by='auc', ascending=False, kind='mergesort')
    result.to_csv(path_or_buf=path, index=False)
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def nrlmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    # max_har_mean = 0
    for r in [50, 100]:
        for x in np.arange(-5, 2):
            for y in np.arange(-5, 3):
                for z in np.arange(-5, 1):
                    for t in np.arange(-3, 1):
                        tic = time.clock()
                        model = NRLMF(cfix=para['c'], K1=para['K1'], K2=para['K2'], num_factors=r, lambda_d=2. ** (x),
                                      lambda_t=2. ** (x), alpha=2. ** (y), beta=2. ** (z), theta=2. ** (t),
                                      max_iter=100)
                        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
                        print(cmd)
                        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                        print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
                            auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))

                        har_mean = 2. / (1. / auc_avg + 1. / aupr_avg)

                        if auc_avg > max_auc:
                            max_auc = auc_avg
                            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
                        # if har_mean > max_har_mean:
                        #     max_har_mean = har_mean
                        #     auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    # max_har_mean = 0
    for x in np.arange(-6, 3):  # [-6, 2]
        for y in np.arange(-6, 3):  # [-6, 2]
            tic = time.clock()
            model = NetLapRLS(gamma_d=10. ** (x), gamma_t=10. ** (x), beta_d=10. ** (y), beta_t=10. ** (y))
            cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
            print(cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
                auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))

            har_mean = 2. / (1. / auc_avg + 1. / aupr_avg)

            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def blmnii_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    # max_har_mean = 0
    for x in np.arange(0, 1.1, 0.1):
        tic = time.clock()
        model = BLMNII(alpha=x, avg=False)
        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
            auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))

        har_mean = 2. / (1. / auc_avg + 1. / aupr_avg)

        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def wnngip_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    # max_har_mean = 0
    for x in np.arange(0.1, 1.1, 0.1):
        for y in np.arange(0.0, 1.1, 0.1):
            tic = time.clock()
            model = WNNGIP(T=x, sigma=1, alpha=y)
            cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
            print(cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
                auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))

            har_mean = 2. / (1. / auc_avg + 1. / aupr_avg)

            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def kbmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for d in [50, 100]:
        tic = time.clock()
        model = KBMF(num_factors=d)
        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
            auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf: %.6f, aupr_conf: %.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def cmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    # for d in [50, 100]:
    #     for x in np.arange(-2, -1):
    #         for y in np.arange(-3, -2):
    #             for z in np.arange(-3, -2):
    # fix it:
    for d in [50, 100]:
        for x in np.arange(-2, 1):
            for y in np.arange(-3, 2):
                for z in np.arange(-3, 2):
                    tic = time.clock()
                    model = CMF(K=d, lambda_l=2. ** (x), lambda_d=2. ** (y), lambda_t=2. ** (z), max_iter=30)
                    cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
                    print(cmd)
                    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                    print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
                        auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))

                    if auc_avg > max_auc:
                        max_auc = auc_avg
                        auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf: %.6f, aupr_conf: %.6f\n" % (
        auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def mdhgi_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for d in np.arange(-4, 3, dtype=float):
        for x in [0.2, 0.4, 0.6, 0.8]:  # np.arange(1, 7, dtype=float)
            tic = time.clock()
            model = MDHGI(alpha=2 ** (d), a=x)
            cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
            print(cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)

            print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
                auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic))
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def cmfmda_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    #
    # for d in [50, 100]:
    #     for x in np.arange(-2, -1):
    #         for y in np.arange(-3, -2):
    #             for z in np.arange(-3, -2):
    # fix it:
    for b in [3, 5, 10]:#K,ita 两个参数论文中未给出范围，这里给几个候选参数
        for c in [0.3, 0.5, 0.7]:
            for d in [50, 100]:
                if d > np.shape(X)[0] or d > np.shape(X)[1]:
                    continue
                for x in np.arange(-2, 1):
                    for y in np.arange(-3, 2):
                        for z in np.arange(-3, 2):
                            tic = time.clock()
                            model = CMFMDA(K=b, ita=c, k_dimen=d, lambda_l=2. ** (x), lambda_d=2. ** (y),
                                           lambda_t=2. ** (z), max_iter=30)
                            cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
                            print(cmd)
                            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                            print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
                                auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))
                            # har_mean = 2. / (1. / auc_avg + 1. / aupr_avg)
                            if auc_avg > max_auc:
                                max_auc = auc_avg
                                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf: %.6f, aupr_conf: %.6f\n" % (
        auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def simclda_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(-3, 1):  #
        for y in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for z in [0.2, 0.4, 0.6, 0.8]:  #
                tic = time.clock()
                model = SIMCLDA(lambda_r=10. ** (x), alpha_l=y, alpha_d=z, max_iter=30)
                cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
                print(cmd)
                aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
                    auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))
                if auc_avg > max_auc:
                    max_auc = auc_avg
                    auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf: %.6f, aupr_conf: %.6f\n" % (
        auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def pbmda_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    Max_length = 3
    for Similarity_threshold in np.arange(0.2, 0.9, 0.1):
        tic = time.clock()
        model = PBMDA(Max_length=Max_length, Pa=2.26, Similarity_threshold=Similarity_threshold)
        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
            auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf: %.6f, aupr_conf: %.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def grnmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    dim = [50, 100]
    L = [2. ** -2, 2. ** -1, 2. ** -0, 2. ** -1]
    M = [0., 10. ** -4, 10. ** -3, 10. ** -2, 10. ** -1]
    Alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    KK = [1, 2, 3, 4, 5]

    par = [[k, l, m, alpha, K] for k in dim for l in L for m in M for alpha in Alpha for K in KK]
    num_par = len(par)

    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists('./result/' + 'cvs' + str(cvs)):
        os.mkdir('./result/' + 'cvs' + str(cvs))
    pklfilename = 'GRNMF_var_' + dataset + '.pkl'
    # weather to  continue the former process
    if (para['cv_continue']) & (os.path.exists(os.path.join('./result/cvs' + str(cvs), pklfilename))):
        with open(os.path.join('./result/cvs' + str(cvs), pklfilename), 'rb') as pklhandler:
            par, max_auc, auc_opt = pickle.load(pklhandler)

    while len(par):
        k, l, m, alpha, K = par[0]
        tic = time.clock()
        model = GRNMF(k=k, lambda_l=l, lambda_m=m, lambda_d=m, r=alpha, K=K, max_iter=30)
        cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
            auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]

        del par[0]
        store_var = [par, max_auc, auc_opt]
        with open(os.path.join('./result/cvs' + str(cvs), pklfilename), 'wb') as pklhandler:
            pickle.dump(store_var, pklhandler)

        cmd = "Present optimal setting:\n%s\n" % auc_opt[0]
        cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Process: %d/%d \n" % (
            auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4], num_par - len(par), num_par)
        print(cmd)

    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf: %.6f, aupr_conf: %.6f\n" % (
        auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def drcc_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for c in [50, 100, 150]:
        for n in np.arange(1, 11, 1):  #
            for miu in [0.1, 1, 10, 100, 500, 1000]:
                m = c
                lam = miu
                tic = time.clock()
                model = DRCC(n=n, c=c, m=m, miu=miu, lam=lam, max_iter=200)
                cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
                print(cmd)
                aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (
                    auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))
                if auc_avg > max_auc:
                    max_auc = auc_avg
                    auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]

    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf: %.6f, aupr_conf: %.6f\n" % (
        auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)
