
from functions import *
import scipy.stats as st
import numpy as np
with open("results_sign2.csv", "w") as resFile:
    for cv in ["1"]: #"1", "2", "3"
        print "CVS:"+cv
        resFile.write("\n")
        resFile.write("CVS;dataset;method;AUC;AUPR;PREC;REC;ACC;t_AUC;p_AUC;t_AUPR;p_AUPR;t_PREC;p_PREC;t_REC;p_REC;t_ACC;p_ACC\n" )
        dt = ["e"]#,"e" #"gpcr","ic", "nr","kinase"
        
        met = ["blmnii", "wnngip", "netlaprls", "aladin"] 
        for dataset in dt: #"gpcr","ic", "nr", "e" , "e"
            resFile.write("\n")
            max_auc = 0
            max_aupr = 0            


            v_max_auc = np.ones(50)
            v_max_aupr = np.ones(50)


            for cp in met: #get maximal values for each evaluation metric throughout the evaluated methods
                v_auc = load_metric_vector("output/"+cp+"_auc_cvs"+cv+"_"+dataset+".txt")[0:21]
                v_aupr = load_metric_vector("output/"+cp+"_aupr_cvs"+cv+"_"+dataset+".txt")[0:21]

                
                avg_auc = np.mean(v_auc)
                if avg_auc > max_auc:
                    max_auc = avg_auc
                    v_max_auc = v_auc[:]
                avg_aupr = np.mean(v_aupr)
                if avg_aupr > max_aupr:
                    max_aupr = avg_aupr
                    v_max_aupr = v_aupr[:]
  
                    
            for cp in met:  #calculate stat. sign. of other methods vs. the best one
                cp_auc = load_metric_vector("output/"+cp+"_auc_cvs"+cv+"_"+dataset+".txt")[0:21]
                cp_aupr = load_metric_vector("output/"+cp+"_aupr_cvs"+cv+"_"+dataset+".txt")[0:21]

                
                x1, y1 = st.ttest_rel(v_max_auc, cp_auc)
                x2, y2 = st.ttest_rel(v_max_aupr, cp_aupr)

                resFile.write("CVS:"+cv+";"+dataset+";"+cp+";%.6f;%.6f;%.9f;%.9f;%.9f;%.9f\n" % (np.mean(cp_auc), np.mean(cp_aupr),  x1, y1, x2, y2) )
                print dataset, cp, np.mean(cp_auc), np.mean(cp_aupr),   x1, y1, x2, y2
            print ""
