
import os
import csv
import numpy as np
import rank_metrics as rank
from functions import *

class newDTIPrediction:
    def __init__(self):        
        with open(os.path.join('data','novelDrugsKEGG.csv'), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='"')
            kg = np.array(list(reader))            
            t = kg[np.arange(1,kg.shape[0]),0]
            d = kg[np.arange(1,kg.shape[0]),1]
            self.kegg = zip(d,t)
        
        with open(os.path.join('data','novelDrugsDrugBank.csv'), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='"')
            db = np.array(list(reader))            
            t = db[np.arange(1,db.shape[0]),1]
            d = db[np.arange(1,db.shape[0]),0]
            self.drugBank = zip(d,t)  
        
        with open(os.path.join('data','novelDrugsMatador.csv'), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='"')
            mt = np.array(list(reader))            
            t = mt[np.arange(1,mt.shape[0]),1]
            d = mt[np.arange(1,mt.shape[0]),0]
            self.matador = zip(d,t)  
                   
                    
        #print(self.kegg)
        #print(self.drugBank)
        #print(self.matador)
        
            
    def verify_novel_interactions(self, method, dataset, sz, predict_num, drug_names, target_names):    
        drugs = np.unique(sz[:,0])
        targets = np.unique(sz[:,1])
        self.precision = 0

        
        new_dti_drugs = os.path.join('output/newDTI', "_".join([method, dataset,str(predict_num), "new_dti.csv"]))
        out_dti_d = open(new_dti_drugs, "w")
        out_dti_d.write(('drug;target;score;hit;kegg_hit;drugBank_hit;matador_hit\n'))


        self.allData = self.kegg + self.drugBank + self.matador
        self.dataset_new_interactions = set([s for s in self.allData if any(s[0] in d for d in drug_names) and any(s[1] in t.replace("hsa","hsa:") for t in target_names)])

        hitSum = 0
        
        for item in sz:
            dti_score = item[2]
            pred_dti = (drug_names[int(item[0])], target_names[int(item[1])], dti_score)             
            hit, kg_hit, db_hit, mt_hit = self.novel_prediction_analysis(pred_dti)     
            hitSum = hitSum + hit
            out_dti_d.write('%s;%s;%f;%i;%i;%i;%i \n'%(pred_dti[0], pred_dti[1], pred_dti[2],hit, kg_hit, db_hit, mt_hit))
            
                       
        print("finished, precision: %f " % (hitSum/(predict_num+0.0)) )  
       
        
    
    def novel_prediction_analysis(self,dti_pair):   
        eval_dti_pairs = []
        hit_list = []

        kg_hit, db_hit, mt_hit, hit = 0,0,0,0
        d, t, score = dti_pair
        dtp = (d,t.replace("hsa","hsa:"))
        #print(dtp)
        if dtp in self.kegg:
            kg_hit = 1
        if dtp in self.drugBank:
            db_hit = 1
        if dtp in self.matador:
            mt_hit = 1
        hit = max(kg_hit,db_hit,mt_hit)
        
        return hit, kg_hit, db_hit, mt_hit
             
