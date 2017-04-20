'''
We base the WNN-GIP implementation on the one from PyDTI project, https://github.com/stephenliu0423/PyDTI, changes were made to the evaluation procedure


[1] van Laarhoven, Twan, Sander B. Nabuurs, and Elena Marchiori. "Gaussian interaction profile kernels for predicting drug-target interaction." Bioinformatics 27.21 (2011): 3036-3043.
[2] van Laarhoven, Twan, and Elena Marchiori. "Predicting drug-target interactions for new drug compounds using a weighted nearest neighbor profile." PloS one 8.6 (2013): e66952.

Default Parameters:
    T = 0.7 (the parameter T in [2])
    sigma = 1.0
    alpha = 0.5
    gamma = 1.0
'''
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

import cv_eval
from functions import *

class WNNGIP:

    def __init__(self, T=0.7, sigma=1, alpha=0.5, gamma=1.0, hyperParamLearn = False):
        self.T = T      # the decay parameter
        self.sigma = sigma  # the regularization parameter
        self.alpha = alpha  # the weight parameter used in combining different kernels
        self.gamma = gamma  # the bandwidth of the GIP kernel
        self.hyperParamLearn = hyperParamLearn

    def preprocess_wnn(self, R, S, train_inx, new_inx, drug=True):
        for d in new_inx:
            ii = np.argsort(S[d, train_inx])[::-1]
            inx = train_inx[ii]
            for i in xrange(inx.size):
                w = self.T**(i)
                if w >= 1e-4:
                    if drug:
                        R[d, :] += w*R[inx[i], :]
                    else:
                        R[:, d] += w*R[:, inx[i]]
                else:
                    break

    def rls_kron_train(self, R, Kd, Kt):
        m, n = R.shape
        ld, vd = np.linalg.eig(Kd)
        lt, vt = np.linalg.eig(Kt)
        vec = ld.reshape((ld.size, 1))*lt.reshape((1, lt.size))
        vec = vec.reshape((1, vec.size))
        x = vec*(1.0/(vec+self.sigma))
        y = np.dot(np.dot(vt.T, R.T), vd)
        y = y.reshape((1, y.size))
        z = (x*y).reshape((n, m))  # need to check
        self.predictR = np.dot(np.dot(vd, z.T), vt.T)

    def kernel_combination(self, R, S, new_inx, bandwidth):
        K = self.alpha*S+(1.0-self.alpha)*rbf_kernel(R, gamma=bandwidth)
        K[new_inx, :] = S[new_inx, :]
        K[:, new_inx] = S[:, new_inx]
        return K

    def learn_hyperparameters(self, intMat, drugMat, targetMat, seed=500):    
        cv_data_optimize_params = cross_validation(intMat, [seed], 1, 0, num=5)
        params = cv_eval.wnngip_cv_eval("wnngip", "","", cv_data_optimize_params, intMat, drugMat, targetMat, 1, "")
        self.T = params["x"]
        self.alpha = params["y"]     

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None, epsilon=0.1):
        R = W*intMat
        if self.hyperParamLearn == False:
            self.learn_hyperparameters( R, drugMat, targetMat)        
        m, n = intMat.shape
        x, y = np.where(R > 0)
        # Enforce the positive definite property of similarity matrix
        drugMat = (drugMat+drugMat.T)/2 + epsilon*np.eye(m)
        targetMat = (targetMat+targetMat.T)/2 + epsilon*np.eye(n)
        train_drugs = np.array(list(set(x.tolist())), dtype=np.int32)
        train_targets = np.array(list(set(y.tolist())), dtype=np.int32)
        new_drugs = np.array(list(set(xrange(m)) - set(x.tolist())), dtype=np.int32)
        new_targets = np.array(list(set(xrange(n)) - set(y.tolist())), dtype=np.int32)
        drug_bw = self.gamma*m/len(x)
        target_bw = self.gamma*n/len(x)
        Kd = self.kernel_combination(R, drugMat, new_drugs, drug_bw)
        Kt = self.kernel_combination(R.T, targetMat, new_targets, target_bw)
        self.preprocess_wnn(R, drugMat, train_drugs, new_drugs, True)
        self.preprocess_wnn(R, targetMat, train_targets, new_targets, False)
        self.rls_kron_train(R, Kd, Kt)

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predictR[inx[:, 0], inx[:, 1]]

    def evaluation(self, test_data, test_label):
        scores = self.predictR[test_data[:, 0], test_data[:, 1]]
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
        return "Model: RLSWNN, T:%s, sigma:%s, alpha:%s, gamma:%s" % (self.T, self.sigma, self.alpha, self.gamma)
    
