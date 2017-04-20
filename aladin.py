'''

Default Parameters:
    alpha = 0.5
    gamma = 1.0 (the gamma0 in [1], see Eq. 11 and 12 for details)
    avg = False (True: g=mean, False: g=max)
    sigma = 1.0 (The regularization parameter used for the RLS-avg classifier)
'''
from __future__ import division
import sys
import numpy as np
import pyhubs
import random
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from scipy.spatial import distance

import cv_eval
from functions import *

class ALADIN:

    def __init__(self, k=3, seedList=[], featureSetSize=-1, model="ECKNN", avg=False, hpLearning = 0, useKNN = 0, hyperParamLearn = False):
        #self.seedList = seedList
        self.seedList = range(1,27)
        self.featureSetSize = int(featureSetSize)
        self.k = int(k)
        self.model = model
        self.hpLearning = hpLearning
        if useKNN == 1:  
          self.model = "KNNreg"
        self.hyperParamLearn = hyperParamLearn
        
        if avg in ('false', 'False', False):
            self.avg = False
        if avg in ('true', 'True', True):
            self.avg = True



    def learn_hyperparameters(self, intMat, drugMat, targetMat, seed=500): 
        cv_data_optimize_params = cross_validation(intMat, [seed], 1, 0, num=5)
        params = cv_eval.aladin_cv_eval("aladin", "","", cv_data_optimize_params, intMat, drugMat, targetMat, 1, "")
        self.k = params["ki"]
        self.featureSetSize = params["features"]
        self.seedList = range(1,params["seeds"])

            
    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        R = W*intMat
        
        if (self.hpLearning == 1) & (self.hyperParamLearn == False):
            self.learn_hyperparameters( R, drugMat, targetMat)
        
        m, n = intMat.shape
        x, y = np.where(R > 0)
        
        self.drugMat = drugMat
        self.targetMat = targetMat
        
        R = R.T
        #i_features = self.intMat[:, list(range(0,d))+list(range(d+1, self.nd))] 
        i_dist_mat = distance.cdist(R, R, 'jaccard') 
        #print (np.shape(i_dist_mat))
        
        #rows 1 and 2 of pseudocode Algorithm 1 of the paper
        i_sim_mat = np.zeros( np.shape(i_dist_mat) )        
        for j in range( np.shape(i_dist_mat)[0] ):
            for j2 in range( np.shape(i_dist_mat)[1] ):
                i_sim_mat[j,j2] = 1-i_dist_mat[j,j2] # jaccard distance -> jaccard similarity
        self.d_features = np.hstack( (self.targetMat, i_sim_mat ) ) 
        
        i_features = np.transpose(R)        
        i_dist_mat = distance.cdist(i_features, i_features, 'jaccard') 
        # drug-drug distances based on their interaction features
        i_sim_mat = np.zeros( np.shape(i_dist_mat) )
        for j in range( np.shape(i_dist_mat)[0] ):
            for j2 in range( np.shape(i_dist_mat)[1] ):
                i_sim_mat[j,j2] = 1-i_dist_mat[j,j2]  # jaccard distance -> jaccard similarity 
        self.t_features = np.hstack( (self.drugMat, i_sim_mat ) ) 
        
        #---------------- create all seeds of features and distance matrices --------------
        self.seeded_t_distmat = defaultdict(list)
        self.seeded_d_distmat = defaultdict(list)
        for sd in self.seedList:
            if self.featureSetSize > -1:
                random.seed(sd)
                selected = random.sample(range(len(self.t_features[0])),self.featureSetSize)
                sd_t_features = self.t_features[:,selected]
                
                random.seed(sd)
                selected = random.sample(range(len(self.d_features[0])),self.featureSetSize)
                sd_d_features = self.d_features[:,selected]
                
            else:
                sd_t_features = self.t_features 
                sd_d_features = self.d_features  
                
            t_dist_mat = distance.cdist(sd_t_features, sd_t_features, 'euclidean')    
            d_dist_mat = distance.cdist(sd_d_features, sd_d_features, 'euclidean')
            
            self.seeded_t_distmat[sd] = t_dist_mat
            self.seeded_d_distmat[sd] = d_dist_mat
            
            
            
            #print "finish distmat sd:%s"%(sd)
            
        self.intMat = R
        self.nd = m
        self.nt = n
        
    def t_dist(self, instance1, instance2):
        return self.t_dist_mat[instance1][instance2]    

    def d_dist(self, instance1, instance2):
        return self.d_dist_mat[instance1][instance2]
    
    def predict_drug(self, sd, d, t):
        """
        if self.featureSetSize > -1:
            random.seed(sd)
            selected = random.sample(range(len(all_features[0])),self.featureSetSize)
            features = all_features[:,selected]
        else:
            features = all_features
            
        self.t_dist_mat = distance.cdist(features, features, 'euclidean')
        """
        
        self.t_dist_mat = self.seeded_d_distmat[sd]
        labels = list(self.intMat[:,d]) # interactions of drug d with all the targets
        train_data = list(range(0,self.nt))

        if all(v==0 for v in labels):
            sum_w = 0
            p = 0
            for k1 in range(self.nd):
                    if k1 == d:
                        continue
                    w = self.drugMat[d][k1] # this is a similarity! (not a distance)
                    p += w*self.intMat[t][k1]
                    sum_w += w
            p /= sum_w
        else:
            """
            if (self.model == "KNNreg"):
                    model1 = pyhubs.KNNreg(k_pred=self.k, metric=self.t_dist)
            if (self.model == "EWKNN"):
                    model1 = pyhubs.EWKNN(k_pred=self.k, k_fit=self.k, metric=self.d_dist)
            if (self.model == "EWCKNN"):
                    model1 = pyhubs.EWCKNN(k_pred=self.k, k_fit=self.k, metric=self.d_dist)
            if (self.model == "ECKNN"):
                    model1 = pyhubs.ECKNN(k_pred=self.k, k_fit=self.k, metric=self.d_dist)			
            model1.fit(train_data, labels)
            """
            
            p = self.d_models[sd][d].predict([t])[0]
            #p = model1.predict([t])[0]
        return p
    
    def predict_target(self, sd, d, t):
        """
        if self.featureSetSize > -1:
            random.seed(sd)
            selected = random.sample(range(len(all_features[0])),self.featureSetSize)
            features = all_features[:,selected]
        else:
            features = all_features

        self.d_dist_mat = distance.cdist(features, features, 'euclidean')
        """
        
        self.d_dist_mat = self.seeded_t_distmat[sd]
        labels = list(self.intMat[t,:]) # interactions of drug d with all the targets       
        train_data = list(range(0,self.nd))

        if all(v==0 for v in labels):
            sum_w = 0
            p = 0
            for k1 in range(self.nt):
                if k1 == t:
                    continue
                w = self.targetMat[t][k1] # this is a similarity
                p +=  w*self.intMat[k1][d]
                sum_w+=w
            p /= sum_w
        else:	
            """
            if (self.model == "KNNreg"):
                model2 = pyhubs.KNNreg(k_pred=self.k, metric=t_dist)
            if (self.model == "EWKNN"):
                model2 = pyhubs.EWKNN(k_pred=self.k, k_fit=self.k, metric=self.t_dist)
            if (self.model == "EWCKNN"):
                model2 = pyhubs.EWCKNN(k_pred=self.k, k_fit=self.k, metric=self.t_dist)
            if (self.model == "ECKNN"):
                model2 = pyhubs.ECKNN(k_pred=self.k, k_fit=self.k, metric=self.t_dist)
            model2.fit(train_data, labels)
            """
            
            p = self.t_models[sd][t].predict([d])[0]
            #p = model2.predict([d])[0]
           
        return p
    
    def predict(self, d, t):
        #test_data are needed to remove unknown (tested) interactions from the similarity computation
        
        #########################################################drug-based prediction
        all_features = self.d_features                              
        pd = [self.predict_drug(sd, d, t) for sd in self.seedList]                   
        p1 = np.mean(pd)     
        
        ######################################target-based prediction
        all_features = self.t_features
        pt = [self.predict_target(sd, d, t) for sd in self.seedList]                
        p2 = np.mean(pt)        
        
        ######################################aggregating predictions
        if self.avg:
            scores = 0.5*(p1+p2)
        else:
            scores = np.maximum(p1, p2)
        return scores

    def learn_models(self, test_data): 
        self.d_models = defaultdict(list)
        self.t_models = defaultdict(list)
        
        dt_list = [list(t) for t in zip(*test_data)]
        drugs = np.unique(dt_list[0])
        targets = np.unique(dt_list[1])

            
        for sd in self.seedList:            
            self.d_models[sd] = defaultdict(list)
            self.t_models[sd] = defaultdict(list)
            
            #---------------- create ECKNN models if there are nonzero entries for drug/target
            self.t_dist_mat = self.seeded_d_distmat[sd]
            self.d_dist_mat = self.seeded_t_distmat[sd]
            
            train_data_t = list(range(0,self.nd))
            train_data_d = list(range(0,self.nt))

            #drug-based models
            for d in drugs:            
                labels = list(self.intMat[:,d]) # interactions of drug d with all the targets
                if not all(v==0 for v in labels):
                    if (self.model == "KNNreg"):
                        model1 = pyhubs.KNNreg(k_pred=self.k, metric=self.t_dist)
                    if (self.model == "EWKNN"):
                        model1 = pyhubs.EWKNN(k_pred=self.k, k_fit=self.k, metric=self.t_dist)
                    if (self.model == "EWCKNN"):
                        model1 = pyhubs.EWCKNN(k_pred=self.k, k_fit=self.k, metric=self.t_dist)
                    if (self.model == "ECKNN"):
                        model1 = pyhubs.ECKNN(k_pred=self.k, k_fit=self.k, metric=self.t_dist)
 			
                    model1.fit(train_data_d, labels)
                    self.d_models[sd][d] = model1                    
                    
            #target-based models
            for t in targets:           
                labels = list(self.intMat[t,:]) # interactions of drug d with all the targets   
                if not all(v==0 for v in labels):      
                    if (self.model == "KNNreg"):
                        model2 = pyhubs.KNNreg(k_pred=self.k, metric=self.d_dist)
                    if (self.model == "EWKNN"):
                        model2 = pyhubs.EWKNN(k_pred=self.k, k_fit=self.k, metric=self.d_dist)
                    if (self.model == "EWCKNN"):
                        model2 = pyhubs.EWCKNN(k_pred=self.k, k_fit=self.k, metric=self.d_dist)
                    if (self.model == "ECKNN"):
                        model2 = pyhubs.ECKNN(k_pred=self.k, k_fit=self.k, metric=self.d_dist)                    
                    #model2 = pyhubs.ECKNN(k_pred=self.k, k_fit=self.k, metric=self.d_dist)
                    model2.fit(train_data_t, labels)
                    self.t_models[sd][t] = model2

     
    def predict_scores(self, test_data, test_label):
    
        self.learn_models(test_data)
        
        scores = []
        for d, t in test_data:
            score = self.predict(d,t)      
            scores.append(score)
            
        return scores
                    
                    
    def evaluation(self, test_data, test_label):
    
        self.learn_models(test_data)
        
        scores = []
        for d, t in test_data:
            score = self.predict(d,t)      
            scores.append(score)
            
        
        self.scores = scores    
        ndcg = 0
        ndcg_inv = 0
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        
        #!!!!we should distinguish here between inverted and not inverted methods nDCGs!!!!
        return aupr_val, auc_val, ndcg, ndcg_inv

    def __str__(self):
        return "Model:ALADIN_k357, k:%s, seeds:%s, features:%s, avg:%s" % (self.k, len(self.seedList), self.featureSetSize, self.avg)
