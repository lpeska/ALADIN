'''
We base the BLM-NII implementation on the one from PyDTI project, https://github.com/stephenliu0423/PyDTI, changes were made to the evaluation procedure

[1] Mei, Jian-Ping, et al. "Drug target interaction prediction by learning from local information and neighbors." Bioinformatics 29.2 (2013): 238-245.
[2] van Laarhoven, Twan, Sander B. Nabuurs, and Elena Marchiori. "Gaussian interaction profile kernels for predicting drug-target interaction." Bioinformatics 27.21 (2011): 3036-3043.

Default Parameters:
    alpha = 0.5
    gamma = 1.0 (the gamma0 in [1], see Eq. 11 and 12 for details)
    avg = False (True: g=mean, False: g=max)
    sigma = 1.0 (The regularization parameter used for the RLS-avg classifier)
'''
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

import cv_eval
from functions import *

class BLMNII:

    def __init__(self, alpha=0.5, gamma=1.0, sigma=1.0, avg=False, hyperParamLearn = False):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.hyperParamLearn = hyperParamLearn
        
        if avg in ('false', 'False', False):
            self.avg = False
        if avg in ('true', 'True', True):
            self.avg = True

    def kernel_combination(self, R, S, new_inx, bandwidth):
        K = self.alpha*S+(1.0-self.alpha)*rbf_kernel(R, gamma=bandwidth)
        K[new_inx, :] = S[new_inx, :]
        K[:, new_inx] = S[:, new_inx]
        return K

    def rls_train(self, R, S, K, train_inx, new_inx):
        Y = R.copy()
        for d in new_inx:
            Y[d, :] = np.dot(S[d, train_inx], Y[train_inx, :])
            x1, x2 = np.max(Y[d, :]), np.min(Y[d, :])
            Y[d, :] = (Y[d, :]-x2)/(x1-x2)
        vec = np.linalg.inv(K+self.sigma*np.eye(K.shape[0]))
        return np.dot(np.dot(K, vec), Y)

    def learn_hyperparameters(self, intMat, drugMat, targetMat, seed=500):    
        cv_data_optimize_params = cross_validation(intMat, [seed], 1, 0, num=5)
        params = cv_eval.blmnii_cv_eval("blmnii", "","", cv_data_optimize_params, intMat, drugMat, targetMat, 1, "")
        self.alpha = params["alpha"]

            
    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        R = W*intMat
        
        if self.hyperParamLearn == False:
            self.learn_hyperparameters( R, drugMat, targetMat)
        
        m, n = intMat.shape
        x, y = np.where(R > 0)
        drugMat = (drugMat+drugMat.T)/2
        targetMat = (targetMat+targetMat.T)/2
        train_drugs = np.array(list(set(x.tolist())), dtype=np.int32)
        train_targets = np.array(list(set(y.tolist())), dtype=np.int32)
        new_drugs = np.array(list(set(xrange(m)) - set(x.tolist())), dtype=np.int32)
        new_targets = np.array(list(set(xrange(n)) - set(y.tolist())), dtype=np.int32)
        drug_bw = self.gamma*m/len(x)
        target_bw = self.gamma*n/len(x)

        Kd = self.kernel_combination(R, drugMat, new_drugs, drug_bw)
        Kt = self.kernel_combination(R.T, targetMat, new_targets, target_bw)
        self.Y1 = self.rls_train(R, drugMat, Kd, train_drugs, new_drugs)
        self.Y2 = self.rls_train(R.T, targetMat, Kt, train_targets, new_targets)

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        x, y = inx[:, 0], inx[:, 1]
        if self.avg:
            scores = 0.5*(self.Y1[x, y]+self.Y2.T[x, y])
        else:
            scores = np.maximum(self.Y1[x, y], self.Y2.T[x, y])
        return scores

    def evaluation(self, test_data, test_label):
        x, y = test_data[:, 0], test_data[:, 1]
        if self.avg:
            scores = 0.5*(self.Y1[x, y]+self.Y2.T[x, y])
        else:
            scores = np.maximum(self.Y1[x, y], self.Y2.T[x, y])
        
        self.scores = scores    
        x, y = test_data[:, 0], test_data[:, 1]
        test_data_T = np.column_stack((y,x))
        
        ndcg = normalized_discounted_cummulative_gain(test_data, test_label, np.array(scores))
        ndcg_inv = normalized_discounted_cummulative_gain(test_data_T, test_label, np.array(scores))
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        
        #!!!!we should distinguish here between inverted and not inverted methods nDCGs!!!!
        return aupr_val, auc_val, ndcg, ndcg_inv

    def __str__(self):
        return "Model:BLMNII, alpha:%s, gamma:%s, sigma:%s, avg:%s" % (self.alpha, self.gamma, self.sigma, self.avg)

