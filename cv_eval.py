
import time
from functions import *
from netlaprls import NetLapRLS
from blmnii import BLMNII
from wnngip import WNNGIP
from aladin import ALADIN

from joblib import Parallel, delayed

def blmnii_cv_eval(method, dataset,output_dir, cv_data, X, D, T, cvs, para):
    max_metric, metric_opt, optArg  = 0, [], []
    for x in np.arange(0, 1.1, 0.1):
        tic = time.clock()
        model = BLMNII(alpha=x, avg=False, hyperParamLearn = True)

        aupr_vec, auc_vec, ndcg_vec , ndcg_inv_vec, results = train(model, cv_data, X, D, T, hyperParamLearn = True)                        
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)

        #print "auc:%.6f, aupr: %.6f, Time:%.6f\n" % (auc_avg, aupr_avg, time.clock()-tic)
        metric = aupr_avg + auc_avg
        if metric > max_metric:
            max_metric = metric
            metric_opt= [ auc_avg, aupr_avg ]
            optArg = {"alpha":x}  
    
    cmd = "Optimal parameter setting:\n" 
    cmd += "alpha: %.6f, auc: %.6f, aupr: %.6f\n" % (optArg["alpha"], metric_opt[0], metric_opt[1])
    print cmd
    return optArg

def aladin_par_eval( ki, seeds, features, cv2, X, D, T):
    t = time.clock()
    model = ALADIN(k=ki, seedList=range(1,seeds), featureSetSize=features, model="ECKNN", avg=False, hyperParamLearn = True)

    aupr_vec, auc_vec, ndcg_vec , ndcg_inv_vec, results = train(model, cv2, X, D, T, hyperParamLearn = True)                        
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    
    metric = aupr_avg + auc_avg
    #print("learning k: %s, seeds: %s, features: %s, metric:  %.6f, time:  %.6f" % (ki, seeds, features, metric, time.clock()-t) ) 
    return metric

def aladin_cv_eval(method, dataset,output_dir, cv_data, X, D, T, cvs, para): 

    cv2 = defaultdict(list)
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            cv2[seed].append((W, test_data, test_label))
            
     
    optArg, argList  = [], []
    for ki in [3,5,7]: #
        for seeds in [27]:#27 #2
            for features in [10,20,50]: #10,20,50   #-1
                argList.append((ki,seeds,features))
    #results = [aladin_par_eval(kj, sd, f, cv2, X, D, T) for kj, sd, f in argList]
    if dataset == "kinase":
        X[X == 0] = -1    
    results = Parallel(n_jobs=3)(delayed(aladin_par_eval)( kj, sd, f, cv2, X, D, T) for kj, sd, f in argList)    
    maxIndex = results.index(max(results))
    ki, seeds, features = argList[maxIndex]
    optArg = {"ki":ki, "seeds":seeds, "features":features}  
                        

    cmd = "Optimal parameter setting:\n"
    cmd += "ki:%s, seeds:%s, features:%s, metric: %.6f\n" % (optArg["ki"], optArg["seeds"], optArg["features"], max(results))
    print cmd
    return optArg



def wnngip_cv_eval(method, dataset,output_dir, cv_data, X, D, T, cvs, para):   
    max_metric, metric_opt, optArg  = 0, [], []
    for x in np.arange(0.1, 1.1, 0.1):
        for y in np.arange(0.0, 1.1, 0.1):
            tic = time.clock()
            model = WNNGIP(T=x, sigma=1, alpha=y, hyperParamLearn = True)

            aupr_vec, auc_vec, ndcg_vec , ndcg_inv_vec, results = train(model, cv_data, X, D, T, hyperParamLearn = True)                        
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)

            metric = aupr_avg + auc_avg
            if metric > max_metric:
                max_metric = metric
                metric_opt= [ auc_avg, aupr_avg]
                optArg = {"x":x, "y":y}   
    
    cmd = "Optimal parameter setting:\n"
    cmd += "x:%.6f, y:%.6f, auc: %.6f, aupr: %.6f\n" % (optArg["x"], optArg["y"], metric_opt[0], metric_opt[1])
    print cmd
    return optArg


def netlaprls_cv_eval(method, dataset,output_dir, cv_data, X, D, T, cvs, para): 
    max_metric, metric_opt, optArg  = 0, [], []
    for x in np.arange(-6, 3):   
        for y in np.arange(-6, 3):  
            tic = time.clock()
            model = NetLapRLS(gamma_d=10**(x), gamma_t=10**(x), beta_d=10**(y), beta_t=10**(y), hyperParamLearn = True)

            aupr_vec, auc_vec, ndcg_vec , ndcg_inv_vec, results = train(model, cv_data, X, D, T, hyperParamLearn = True)                        
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)

            metric = aupr_avg + auc_avg
            if metric > max_metric:
                max_metric = metric
                metric_opt= [auc_avg, aupr_avg]
                optArg = {"x":10**(x), "y":10**(y)}         
    
    cmd = "Optimal parameter setting:\n%s\n" % metric_opt[0]
    cmd += "x:%.6f, y:%.6f, auc: %.6f, aupr: %.6f\n" % (optArg["x"], optArg["y"], metric_opt[0], metric_opt[1])
    print cmd
    return optArg    
    
    
def cmf_cv_eval(method, dataset,output_dir, cv_data, X, D, T, cvs, para):

    cv2 = defaultdict(list)
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            cv2[seed].append((W, test_data, test_label))
            break
         
    max_metric, metric_opt, optArg  = 0, [], []
    for d in [100]:
        for x in np.arange(-2, 1):
            for y in np.arange(-5, -2):
                for z in np.arange(-5, -2):
                    tic = time.clock()
                    model = CMF(K=d, lambda_l=2**(x), lambda_d=2**(y), lambda_t=2**(z), max_iter=100, hyperParamLearn = True)

                    aupr_vec, auc_vec, ndcg_vec , ndcg_inv_vec, results = train(model, cv2, X, D, T, hyperParamLearn = True)                        
                    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                    auc_avg, auc_conf = mean_confidence_interval(auc_vec)

                    metric = aupr_avg + auc_avg
                    if metric > max_metric:
                        max_metric = metric
                        metric_opt= [auc_avg, aupr_avg]
                        optArg = {"d":d, "x":2**(x), "y":2**(y), "z":2**(z)}   

    
    cmd = "Optimal parameter setting:\n%s\n" % metric_opt[0]
    cmd += "d:%.6f, x:%.6f, y:%.6f, z:%.6f, auc: %.6f, aupr: %.6f\n" % (optArg["d"], optArg["x"], optArg["y"], optArg["z"], metric_opt[0], metric_opt[1])
    print cmd
    return optArg
