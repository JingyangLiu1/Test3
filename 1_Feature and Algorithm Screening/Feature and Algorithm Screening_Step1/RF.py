#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from itertools import combinations
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import RFE
from bayes_opt import BayesianOptimization

data=pd.read_excel(r"C:\Users\HP\Desktop\Data.xlsx",sheet_name='16+3',index_col=0,header=0)
data1=data.iloc[0:16, :]

minmax_scaler=preprocessing.MinMaxScaler()  
data2=minmax_scaler.fit_transform(data1)
data3=pd.DataFrame(data2,columns=['O3_input','lg(O3)','H2O2_input','lg(H2O2)','H2O2/O3','lg(H2O2/O3)','pH','volumn',
                                  'TOC','UV254','Φn1','Φn2','Φn3','Φn4','Φn5','FRI','FRI20','OU20','OU60','OU120',
                                 'O3(1)20','O3(1)60','O3(1)120','O3(g)20','O3(g)60','O3(g)120','H2O2_20','H2O2_60',
                                  'H2O2_120','FMax1','FMax2','FMax3','FMax4','cost'])
data4=data3[['lg(O3)','lg(H2O2)','pH','TOC','UV254','Φn1','Φn2','Φn3','Φn4','Φn5','FRI','OU20','OU60','OU120',
                                 'O3(1)20','O3(1)60','O3(1)120','O3(g)20','O3(g)60','O3(g)120','H2O2_20','H2O2_60',
                                  'H2O2_120','FMax1','FMax2','FMax3','FMax4']]

def MODEL(X, y):

    def black_box_function(n_estimators, min_samples_split, max_features, max_depth,max_leaf_nodes):                                                                          
        model= RandomForestRegressor(n_estimators=int(n_estimators),
                                min_samples_split=int(min_samples_split),
                                max_features=min(max_features, 0.999),  
                                max_depth=int(max_depth),
                                max_leaf_nodes=int(max_leaf_nodes),
                                random_state=2)
        loo = LeaveOneOut()
        y_real = []
        y_predicted = []

        # Applying LOO-CV
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train.ravel())
            y_pred = model.predict(X_test)
            y_real.append(y_test[0])
            y_predicted.append(y_pred[0])
        res = r2_score(y_real, y_predicted)
        return res
    
    pbounds= {'n_estimators': (10, 500),
         'min_samples_split': (2, 25),
         'max_features': (1, 4),
         'max_depth': (1, 5),
         'max_leaf_nodes':(2,15)}
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1)
    optimizer.maximize(
        init_points=20, 
        n_iter=30)
    maxx  = optimizer.max
    maxx1 = maxx['params']
    maxx2 =maxx['target'] 

    return maxx2

data7 = data4.values

column_names = data4.columns

X_original = data7[:, [i for i in range(data7.shape[1]) if i!= 3]]
y_original = data7[:, 3]

initial_features = X_original[:, :3]

all_results = []

for comb in combinations(range(3, X_original.shape[1]), 2):

    new_feature_subset = np.hstack((initial_features,
                                    X_original[:, comb[0]].reshape(-1, 1),
                                    X_original[:, comb[1]].reshape(-1, 1)))

    maxx2 = MODEL(new_feature_subset, y_original)

    all_results.append((maxx2, new_feature_subset, [0, 1, 2] + list(comb)))

all_results.sort(key=lambda x: x[0], reverse=True)

# Outputs the top 5 highest maxx2 values, corresponding feature subsets and feature indexes.
for i in range(5):
    print(f"top {i + 1}maxx2 value:", all_results[i][0])

    data7_indexes = [index if index < 3 else index + 1 for index in all_results[i][2]]

    print(f"The corresponding feature subset:", all_results[i][1])
    print(f"The corresponding index in data7:", data7_indexes)
    print(f"The corresponding variable name:", [column_names[index] for index in data7_indexes])

