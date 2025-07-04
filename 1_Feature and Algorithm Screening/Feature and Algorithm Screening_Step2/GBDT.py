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
data1=data.iloc[0:18, :]    
OP=data1[['lg(O3)','lg(H2O2)','pH','TOC','Φn4','O3(l)20']]  
minmax_scaler=preprocessing.MinMaxScaler()  
data2=minmax_scaler.fit_transform(OP)
data3=pd.DataFrame(data2,columns=['lg(O3)','lg(H2O2)','pH','TOC','Φn4','O3(l)20'])  

X_full=data3[['lg(O3)','lg(H2O2)','pH','Φn4','O3(l)20']]  
y_full=data3['TOC']

def run_bayesian_optimization(X_train_all, y_train_all):
    def black_box_function(learning_rate, n_estimators, min_samples_split, max_features, max_depth, max_leaf_nodes):
        model = GradientBoostingRegressor(
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            max_depth=int(max_depth),
            max_leaf_nodes=int(max_leaf_nodes),
            random_state=2
        )
        loo = LeaveOneOut()
        y_real, y_predicted = [], []
        for train_index, test_index in loo.split(X_train_all):
            X_train, X_val = X_train_all[train_index], X_train_all[test_index]
            y_train, y_val = y_train_all[train_index], y_train_all[test_index]
            model.fit(X_train, y_train.ravel())
            y_pred = model.predict(X_val)
            y_real.append(y_val[0])
            y_predicted.append(y_pred[0])
        return r2_score(y_real, y_predicted)
    
    pbounds = {
        'learning_rate': (0.001, 0.2),
        'n_estimators': (10, 500),
        'min_samples_split': (2, 25),
        'max_features': (1, 4),
        'max_depth': (1, 5),
        'max_leaf_nodes': (2, 15)
    }
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1
    )
    optimizer.maximize(init_points=15, n_iter=20)
    return optimizer.max['params']


test_scores = []
test_rmse_scores = []
n_runs = 10  

for run in range(n_runs):
    
    X_train_all, X_test, y_train_all, y_test = train_test_split(
    X_full.values,  
    y_full.values, 
    test_size=4, 
    random_state=run
)
    
    best_params = run_bayesian_optimization(X_train_all, y_train_all)
    
    final_model = GradientBoostingRegressor(
        learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        max_leaf_nodes=int(best_params['max_leaf_nodes']),
        max_features=min(best_params['max_features'], 0.999),
        min_samples_split=int(best_params['min_samples_split']),
        max_depth=int(best_params['max_depth']),
        random_state=2
    )
    final_model.fit(X_train_all, y_train_all.ravel())
    y_pred = final_model.predict(X_test)

    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
    
    test_scores.append(test_r2)
    test_rmse_scores.append(test_rmse)  
    
    print(f"Run {run+1}/10 - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")  


mean_r2 = np.mean(test_scores)
std_r2 = np.std(test_scores)
mean_rmse = np.mean(test_rmse_scores)  
std_rmse = np.std(test_rmse_scores)    

print(f"\n Average test R²score: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"verage testRMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

