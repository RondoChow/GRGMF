## Create this copy to test the contrain of adjacency Matrix of similarity
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import getopt
import cv_eval
from functions import *
from models import ADPGMF, NRLMF #, NetLapRLS, BLMNII, WNNGIP, KBMF, CMF, MDHGI, CMFMDA, SIMCLDA, PBMDA, GRNMF, DRCC
from new_pairs import novel_prediction_analysis, MDA_novel_prediction_analysis


import pandas as pd
import logging

#TODO@Alfred(20191016): replace all print with logging
logging.basicConfig(level='WARNING')


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:d:f:c:s:o:n:M:p", ["method=", "dataset=", "data-dir=", "cvs=",
                                                               "specify-arg=", "method-options=", "predict-num=",
                                                               "MDA=", "output-dir=", ])
    except getopt.GetoptError:
        sys.exit()

    data_dir = "E:/GRGMF/GRGMF"#os.path.join(os.path.pardir)
    output_dir = "E:/GRGMF/GRGMF/output/"#os.path.join(os.path.pardir, 'output')
    cvs, sp_arg, model_settings, predict_num = 1, 1, [], 0

    seeds = [7771, 8367, 22, 1812, 4659]
    # seeds = np.random.choice(10000, 5, replace=False)
    MDA = None
    for opt, arg in opts:
        if opt == "--method":
            method = arg
        if opt == "--dataset":
            dataset = arg
        if opt == "--data-dir":
            data_dir = arg
        if opt == "--output-dir":
            output_dir = arg
        if opt == "--cvs":
            cvs = int(arg)
        if opt == "--specify-arg":
            sp_arg = int(arg)
        if opt == "--method-options":
            model_settings = [s.split('=') for s in str(arg).split()]
        if opt == "--predict-num":
            predict_num = int(arg)
        if opt == "--MDA":
            MDA = arg
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # default parameters for each methods
    # K1 is the size of neighbors
    if method == 'nrlmf':
        args = {'c': 5, 'K1': 5, 'K2': 5, 'mf_dim': 60, 'lambda_d': 0.125, 'lambda_t': 0.125, 'alpha': 0.25,
                'beta': 0.125, 'theta': 0.5, 'max_iter': 100}
    if method == 'netlaprls':
        args = {'gamma_d': 10, 'gamma_t': 10, 'beta_d': 1e-5, 'beta_t': 1e-5}
    if method == 'blmnii':
        args = {'alpha': 0.7, 'gamma': 1.0, 'sigma': 1.0, 'avg': False}
    if method == 'wnngip':
        args = {'T': 0.8, 'sigma': 1.0, 'alpha': 0.8}
    if method == 'kbmf':
        args = {'num_factors': 50}
    if method == 'cmf':
        args = {'K': 50, 'lambda_l': 0.5, 'lambda_d': 0.125, 'lambda_t': 0.125, 'max_iter': 30}
    if method == 'neumf':
        args = {'K': 5, 'max_iter': 100, 'num_negative': 2, 'lr': 0.003, 'batch_size': 128, 'mf_dim': 8,
                'layers': [64, 32, 16, 8], 'reg_layers': [0, 0, 0, 0], 'reg_mf': 0, 'evl_process': 1,
                'encode_l1regu': False}
    if method == 'gmf':
        args = {'K': 5, 'max_iter': 100, 'num_negative': 2, 'lr': 0.003, 'batch_size': 128, 'mf_dim': 8,
                'reg_mf': [0, 0], 'evl_process': 1, 'input_type': 'index'}
    if method == 'inmf':
        args = {'K': 5, 'max_iter': 100, 'num_negative': 2, 'lr': 0.003, 'batch_size': 128, 'mf_dim': 8,
                'layers': [64, 32, 16, 8], 'reg_layers': [0, 0, 0, 0], 'reg_mf': 0, 'evl_process': 1,
                'encode_l1regu': False}
    if method == 'adpgmf':
        args = {'K': 5, 'max_iter': 200, 'lr': 0.01, 'lamb': 1.35, 'beta': 0.5, 'r1': 2., 'r2': 4.,
                'mf_dim': 40, 'c': 5, 'resample': 0, 'ita': 0.5, 'cvs': cvs, 'cv_continue': False, 'verpose': 10}
    if method == 'mdhgi':
        args = {'alpha': 0.1, 'a': 0.4}
    if method == 'cmfmda':
        args = {'K': 5, 'ita': 0.5, 'k_dimen': 10, 'lambda_l': 0.5, 'lambda_d': 0.125, 'lambda_t': 0.125,
                'max_iter': 30}
    if method == 'simclda':
        args = {'lambda_r': 0.1, 'alpha_l': 0.8, 'alpha_d': 0.6, 'max_iter': 100}
    if method == 'pbmda':
        args = {'Max_length': 3, 'Pa': 2.26, 'Similarity_threshold': 0.5}
    if method == 'grnmf':
        args = {'k': 50, 'lambda_l': 0.5, 'lambda_m': 0.5, 'lambda_d': 0.5, 'max_iter': 200, 'cv_continue': False,}
    if method == 'drcc':
        args = {'n': 5, 'c': 50, 'm': 50, 'lam': 0.1, 'miu': 0.1, 'max_iter': 200}
    for key, val in model_settings:
        args[key] = eval(val)
    '''
    intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))
    drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir, 'datasets'))
    '''
    intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir))
    drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir))

    if predict_num == 0:
        if cvs == 1:  # CV setting CVS1
            X, D, T, cv = intMat, drugMat, targetMat, 1
        if cvs == 2:  # CV setting CVS2
            X, D, T, cv = intMat, drugMat, targetMat, 0
        if cvs == 3:  # CV setting CVS3
            X, D, T, cv = intMat.T, targetMat, drugMat, 0
        cv_data = cross_validation(X, seeds, cv, num=5)

    if sp_arg == 0 and predict_num == 0:
        if method == 'nrlmf':
            cv_eval.nrlmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'netlaprls':
            cv_eval.netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'blmnii':
            cv_eval.blmnii_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'wnngip':
            cv_eval.wnngip_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'kbmf':
            cv_eval.kbmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'cmf':
            cv_eval.cmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'adpgmf':
            cv_eval.adpgmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'mdhgi':
            cv_eval.mdhgi_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'cmfmda':
            cv_eval.cmfmda_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'simclda':
            cv_eval.simclda_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'pbmda':
            cv_eval.pbmda_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'grnmf':
            cv_eval.grnmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'drcc':
            cv_eval.drcc_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
    if sp_arg == 1 or predict_num > 0:
        tic = time.clock()
        if method == 'nrlmf':
            model = NRLMF(cfix=args['c'], K1=args['K1'], K2=args['K2'], num_factors=args['mf_dim'],
                          lambda_d=args['lambda_d'], lambda_t=args['lambda_t'], alpha=args['alpha'], beta=args['beta'],
                          theta=args['theta'], max_iter=args['max_iter'])
        if method == 'netlaprls':
            model = NetLapRLS(gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], beta_d=args['beta_t'],
                              beta_t=args['beta_t'])
        if method == 'blmnii':
            model = BLMNII(alpha=args['alpha'], gamma=args['gamma'], sigma=args['sigma'], avg=args['avg'])
        if method == 'wnngip':
            model = WNNGIP(T=args['T'], sigma=args['sigma'], alpha=args['alpha'])
        if method == 'kbmf':
            model = KBMF(num_factors=args['num_factors'])
        if method == 'cmf':
            model = CMF(K=args['K'], lambda_l=args['lambda_l'], lambda_d=args['lambda_d'], lambda_t=args['lambda_t'],
                        max_iter=args['max_iter'])
        if method == 'adpgmf':
            model = ADPGMF(max_iter=args['max_iter'], c=args['c'], resample=args['resample'], lamb=args['lamb'],
                           beta=args['beta'], r1=args['r1'], r2=args['r2'], lr=args['lr'],
                           mf_dim=args['mf_dim'],
                           ita=args['ita'], K=args['K'], cvs=args['cvs'], verpose=args['verpose'])
        if method == 'mdhgi':
            model = MDHGI(alpha=args['alpha'], a=args['a'])
        if method == 'cmfmda':
            model = CMFMDA(K=args['K'], ita=args['ita'], k_dimen=args['k_dimen'], lambda_l=args['lambda_l'],
                           lambda_d=args['lambda_d'], lambda_t=args['lambda_t'], max_iter=args['max_iter'])
        if method == 'simclda':
            model = SIMCLDA(lambda_r=args['lambda_r'], alpha_l=args['alpha_l'], alpha_d=args['alpha_d'],
                            max_iter=args['max_iter'])
        if method == 'pbmda':
            model = PBMDA(Max_length=args['Max_length'], Pa=args['Pa'],
                          Similarity_threshold=args['Similarity_threshold'])
        if method == 'grnmf':
            model = GRNMF(k=args['k'], lambda_l=args['lambda_l'], lambda_m=args['lambda_m'], lambda_d=args['lambda_d'],
                          max_iter=args['max_iter'])
        if method == 'drcc':
            model = DRCC(n=args['n'], c=args['c'], m=args['m'], lam=args['lam'], miu=args['miu'],
                         max_iter=args['max_iter'])

        cmd = str(model)
        if predict_num == 0:
            print(("Dataset:" + dataset + " CVS:" + str(cvs) + "\n" + cmd))
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)

            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print(("Dataset:" + dataset + " CVS:" + str(cvs) + "\n" + cmd))
            print(("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" % (
                auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic)))
            write_metric_vector_to_file(auc_vec, os.path.join(output_dir,
                                                              method + "_auc_cvs" + str(cvs) + "_" + dataset + ".txt"))
            #write_metric_vector_to_file(aupr_vec, os.path.join(output_dir, method + "_aupr_cvs" + str(cvs) + "_" + dataset + ".txt"))
            return auc_avg, aupr_avg
        elif predict_num > 0:
            print(("Dataset:" + dataset + "\n" + cmd))
            seed = 7771 if method == 'cmf' else 22

            # #debug
            W_test = np.ones(intMat.shape)
            model.fix_model(W_test, intMat, drugMat, targetMat, seed)

            # model.fix_model(intMat, intMat, drugMat, targetMat, seed)
            x, y = np.where(intMat == 0)
            scores = model.predict_scores(list(zip(x, y)), 5)
            ii = np.argsort(scores)[::-1]
            assert (len(drug_names), len(target_names)) == intMat.shape
            if MDA:
                predict_pairs = [(drug_names[x[i]], target_names[y[i]], scores[i]) for i in ii]
                novel_pairs = pd.DataFrame(predict_pairs)
                novel_pairs.columns = ['MiRNA', 'Disease', 'Probability']
                # novel_pairs.to_excel('./Doc/novel_MDA.xlsx')
                novel_pairs = novel_pairs[novel_pairs['Disease'] == MDA]
                novel_pairs = novel_pairs.iloc[0:min(predict_num, len(novel_pairs))]

                new_mda_file = os.path.join(output_dir, "_".join([method, dataset, MDA, "new_mda.txt"]))
                MDA_novel_prediction_analysis(novel_pairs, new_mda_file, MDA, os.path.join(data_dir, 'MDA_database'))
            else:
                predict_pairs = [(drug_names[x[i]], target_names[y[i]], scores[i]) for i in ii[:predict_num]]
                new_dti_file = os.path.join(output_dir, "_".join([method, dataset, "new_dti.txt"]))
                novel_pairs = pd.DataFrame(predict_pairs)
                novel_pairs.columns = ['drug_name', 'target_name', 'score']
                novel_pairs.to_csv(path_or_buf=new_dti_file, index=False, )
                novel_prediction_analysis(predict_pairs, new_dti_file, os.path.join(data_dir, 'DTI_database'))


if __name__ == "__main__":
    main(sys.argv[1:])
