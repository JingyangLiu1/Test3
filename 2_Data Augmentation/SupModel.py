#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import Normalize
from itertools import combinations
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import RFE
from bayes_opt import BayesianOptimization
from sklearn.inspection import PartialDependenceDisplay

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
features = data[['lg(O3)', 'lg(H2O2)', 'pH', 'TOC', 'FMax2']]
data1 = features.iloc[0:18]

df_GAN_origin_all = pd.read_excel(
    r"C:\Users\HP\jupyternotebook\MLofCC\GAN\GAN800.xlsx",
    header=0)
df_GAN_origin = df_GAN_origin_all[['lg(O3)', 'lg(H2O2)', 'pH', 'TOC']]

scaler1 = MinMaxScaler()
feature_columns = ['lg(O3)', 'lg(H2O2)', 'pH', 'TOC','FMax2']
scaled_features = scaler1.fit_transform(data1[feature_columns])
data3 = pd.DataFrame(scaled_features, columns=feature_columns)


X = data3[['lg(O3)', 'lg(H2O2)', 'pH', 'FMax2']].values
y = data3['TOC'].values

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
    preds, truths = [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        preds.append(model.predict(X_test)[0])
        truths.append(y_test[0])
    return r2_score(truths, preds)

optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds={
            'learning_rate': (0.001, 0.2),
            'n_estimators': (10, 500),
            'min_samples_split': (2, 25),
            'max_features': (1, 4),
            'max_depth': (1, 5),
            'max_leaf_nodes': (2, 15)
        },
        random_state=42
    )
optimizer.maximize(init_points=15, n_iter=25)
best_params = optimizer.max['params']
    
model = GradientBoostingRegressor(
        learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        max_leaf_nodes=int(best_params['max_leaf_nodes']),
        max_features=min(best_params['max_features'], 0.999),
        min_samples_split=int(best_params['min_samples_split']),
        max_depth=int(best_params['max_depth']),
        random_state=2
    )
model.fit(X, y)

model_FMax2 = LinearRegression()
model_FMax2.fit(data3[['lg(O3)']], data3['FMax2'])
    
scaler2 = MinMaxScaler()
train_features = data1[['lg(O3)', 'lg(H2O2)', 'pH', 'TOC']].values  
scaler2.fit(train_features)
gan_features = scaler2.transform(df_GAN_origin[['lg(O3)', 'lg(H2O2)', 'pH', 'TOC']])
df_GAN = pd.DataFrame(gan_features, columns=['lg(O3)', 'lg(H2O2)', 'pH', 'TOC'])
    
df_GAN['FMax2'] = model_FMax2.predict(df_GAN[['lg(O3)']])

X_new = df_GAN[['lg(O3)', 'lg(H2O2)', 'pH', 'FMax2']]
y_pred_new = model.predict(X_new)

toc_min = scaler2.data_min_[-1]
toc_max = scaler2.data_max_[-1]
y_pred_orig = y_pred_new * (toc_max - toc_min) + toc_min

# ================== 0.05，36 ==================
all_train_true = []
all_train_pred = []
all_test_true = []
all_test_pred = []

r2_results = []
rmse_results = []

scaler3 = MinMaxScaler()
feature_columns3 = ['lg(O3)', 'lg(H2O2)', 'pH']
scaler3.fit(data1[feature_columns3])
# ================== Training cycle ==================
for iter in range(10):
    print(f"\n========== Iteration times {iter+1}/10 ==========")
    
    valid_indices = []
    for i in range(len(y_pred_orig)):
        if abs(df_GAN_origin.iloc[i]['TOC'] - y_pred_orig[i]) < 0.05:
            valid_indices.append(i)
    np.random.seed(iter)  

    if len(valid_indices) == 0:
        raise ValueError("No valid samples were found, please adjust the screening threshold.")

    try:
        selected_indices = np.random.choice(valid_indices, size=36, replace=False)
    except ValueError:
        selected_indices = np.random.choice(valid_indices, size=36, replace=True)

    selected_data = df_GAN_origin.iloc[selected_indices]


    test_indices = np.random.choice(data1.index, size=4, replace=False)
    test_set = data1.loc[test_indices]
    train_set = data1.drop(test_indices)
    

    combined_data = pd.concat([train_set, selected_data])
    X_final = scaler3.transform(combined_data[['lg(O3)', 'lg(H2O2)', 'pH']])
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

    train_pred = final_model.predict(X_final)
    all_train_true.extend(y_final.ravel().tolist())
    all_train_pred.extend(train_pred.tolist())

    X_test = scaler3.transform(test_set[['lg(O3)', 'lg(H2O2)', 'pH']])
    y_test = test_set['TOC'].values
    test_pred = final_model.predict(X_test)
    all_test_true.extend(y_test.tolist())
    all_test_pred.extend(test_pred.tolist())
    
    r2 = r2_score(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    r2_results.append(r2)
    rmse_results.append(rmse)
    print(f"Results of this round: R2={r2:.4f}, RMSE={rmse:.4f}")

print("\n================ Final result ================")
print(f"Average R2 score: {np.mean(r2_results):.4f} ± {np.std(r2_results):.4f}")
print(f"Average RMSE score: {np.mean(rmse_results):.4f} ± {np.std(rmse_results):.4f}")
print("\n Detailed R2 results:", [round(x, 4) for x in r2_results])
print("Detailed RMSE results:", [round(x, 4) for x in rmse_results])

