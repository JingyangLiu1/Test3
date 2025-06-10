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
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import RFE
from bayes_opt import BayesianOptimization
from sklearn.inspection import PartialDependenceDisplay
from deap import algorithms, base, creator, tools

combined_data=pd.read_excel(r"C:\Users\HP\jupyternotebook\MLofCC\combined_data.xlsx",header=0)
def run_bayesian_optimization(X_train_all, y_train_all):
    def black_box_function(learning_rate, n_estimators, min_samples_split, max_features, max_depth, max_leaf_nodes):
        params = {
            'learning_rate': max(learning_rate, 1e-3),
            'n_estimators': int(n_estimators),
            'min_samples_split': int(min_samples_split),
            'max_features': min(max_features, 0.999),
            'max_depth': int(max_depth),
            'max_leaf_nodes': int(max_leaf_nodes),
            'random_state': 2
        }
        
        model = GradientBoostingRegressor(**params)
        loo = LeaveOneOut()
        preds, truths = [], []
        
        for train_idx, val_idx in loo.split(X_train_all):
            X_train, X_val = X_train_all[train_idx], X_train_all[val_idx]
            y_train, y_val = y_train_all[train_idx], y_train_all[val_idx]
            model.fit(X_train, y_train.ravel())
            preds.append(model.predict(X_val)[0])
            truths.append(y_val[0])
            
        return r2_score(truths, preds)

    pbounds = {
        'learning_rate': (0.001, 0.2),
        'n_estimators': (10, 500),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 1.0),
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

data=pd.read_excel(r"C:\Users\HP\Desktop\Data.xlsx",sheet_name='16+3',index_col=0,header=0)
data1=data.iloc[0:18, :]
scaler = preprocessing.MinMaxScaler()
feature_columns = ['lg(O3)', 'lg(H2O2)', 'pH']
scaler.fit(data1[feature_columns])
min_value = scaler.data_min_
max_value = scaler.data_max_
min_H2O2=min_value[1]
min_O3=min_value[0]
min_pH=min_value[2]
max_H2O2=max_value[1]
max_O3=max_value[0]
max_pH=max_value[2]

X_final = scaler.transform(combined_data[['lg(O3)', 'lg(H2O2)', 'pH']])
y_final = combined_data['TOC'].values.reshape(-1, 1)
best_params = run_bayesian_optimization(X_final, y_final)
final_model = GradientBoostingRegressor(learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        min_samples_split=int(best_params['min_samples_split']),
        max_features=best_params['max_features'],
        max_depth=int(best_params['max_depth']),
        max_leaf_nodes=int(best_params['max_leaf_nodes']),
        random_state=2)
final_model.fit(X_final, y_final.ravel())

SLY_data=pd.read_excel(r"C:\Users\HP\Desktop\Data.xlsx",index_col=0,header=0)
TransData=SLY_data[['lg(O3)','lg(H2O2)','pH','TOC']]
X_SLY = scaler.transform(TransData[feature_columns])
y_pre_SLY=final_model.predict(X_SLY)
trans = pd.DataFrame(y_pre_SLY, columns=['trans'])
TransData['trans']=trans.values

