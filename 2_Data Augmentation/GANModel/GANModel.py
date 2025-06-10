#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import shap
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from itertools import combinations
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler
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

warnings.filterwarnings('ignore')

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

data = pd.read_excel(r"C:\Users\HP\Desktop\Data.xlsx", 
                    sheet_name='16+3',
                    index_col=0,
                    header=0)
features = data[['lg(O3)', 'lg(H2O2)', 'pH', 'TOC']]
data1 = features.iloc[0:18]

GAN0 = pd.read_excel(r"C:\Users\HP\jupyternotebook\FSL-Github\2_Data Augmentation\GANModel\BS36.xlsx", 
                   header=0)
GAN=GAN0[['lg(O3)', 'lg(H2O2)', 'pH', 'TOC']]

r2_results = []
rmse_results = []

for iter in range(10):
    print(f"\n========== Iteration times {iter+1}/10 ==========")
    
    test_indices = np.random.choice(data1.index, size=4, replace=False)
    test_set = data1.loc[test_indices]
    train_set = data1.drop(test_indices)
    
    scaler = MinMaxScaler()
    
    X_train = scaler.fit_transform(train_set[['lg(O3)', 'lg(H2O2)', 'pH']])
    y_train = train_set['TOC'].values.reshape(-1, 1)
    
    GAN_features = scaler.transform(GAN[['lg(O3)', 'lg(H2O2)', 'pH']])
    GAN_processed = np.hstack([GAN_features, GAN['TOC'].values.reshape(-1, 1)])
    
    X_augmented = np.vstack([X_train, GAN_features])
    y_augmented = np.vstack([y_train, GAN['TOC'].values.reshape(-1, 1)])
    
    best_params = run_bayesian_optimization(X_augmented, y_augmented)
    
    final_model = GradientBoostingRegressor(
        learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        min_samples_split=int(best_params['min_samples_split']),
        max_features=best_params['max_features'],
        max_depth=int(best_params['max_depth']),
        max_leaf_nodes=int(best_params['max_leaf_nodes']),
        random_state=2
    )
    final_model.fit(X_augmented, y_augmented.ravel())
    
    X_test = scaler.transform(test_set[['lg(O3)', 'lg(H2O2)', 'pH']])
    y_test = test_set['TOC'].values
    y_pred = final_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    r2_results.append(r2)
    rmse_results.append(rmse)
    print(f"Results of this round: R2={r2:.4f}, RMSE={rmse:.4f}")

print("\n================ Final result ================")
print(f"Average R2 score: {np.mean(r2_results):.4f} ± {np.std(r2_results):.4f}")
print(f"Average RMSE score: {np.mean(rmse_results):.4f} ± {np.std(rmse_results):.4f}")
print("\n Detailed R2 results:", [round(x, 4) for x in r2_results])
print("Detailed RMSE results:", [round(x, 4) for x in rmse_results])

