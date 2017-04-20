# ALADIN
ALADIN: Advanced Local Drug-Target Interaction Prediction technique

version 1.0, April 20, 2017

--------
This package is written by:

Krisztian Buza and Ladislav Peska
Email: peska@ksi.mff.cuni.cz

-------
This package is based on the PyDTI package by Yong Liu,
https://github.com/stephenliu0423/PyDTI
and PyHubs package available from
http://biointelligence.hu/pyhubs/

--------
ALADIN works on Python 2.7 (tested on Intel Python 2.7.12).
--------
ALADIN requires NumPy, scikit-learn and SciPy to run.
To get the results of different methods, please run eval.py by setting suitable values for the following parameters:

	--method 			set DTI prediction method, i.e. blmnii, netlaprls, wnngip, aladin
	--dataset: 			choose the benchmark dataset, i.e., nr, gpcr, ic, e, kinase
	--method-opt:		set arguments for each method
	--predict-num:		a positive integer for predicting top-N novel DTIs for each drug and target (default 0)
        
	Some examples are in the end of eval.py file

The results can be analysed via result_sign_analysis.py.
