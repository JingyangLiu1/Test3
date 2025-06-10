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

scaler1 = MinMaxScaler()
feature_columns = ['lg(O3)', 'lg(H2O2)', 'pH','FMax2']
scaled_features = scaler1.fit_transform(data1[feature_columns])
data3 = pd.DataFrame(scaled_features, columns=feature_columns)
data3['TOC'] = data1['TOC'].values


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
optimizer.maximize(init_points=20, n_iter=30)
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

def reverse_minmax_normalization(data, min_value, max_value):
    return data * (max_value - min_value) + min_value

model_FMax2 = LinearRegression()
model_FMax2.fit(data3[['lg(O3)']], data3['FMax2'])

df_GAN_origin_all = pd.read_excel(
    r"C:\Users\HP\jupyternotebook\MLofCC\\GAN\GAN800.xlsx",
    header=0)
df_GAN_origin = df_GAN_origin_all[['lg(O3)', 'lg(H2O2)', 'pH', 'TOC']]

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

all_pdp1 = []
all_pdp2 = []
all_pdp3 = []
grids = []

scaler3 = MinMaxScaler()
feature_columns3 = ['lg(O3)', 'lg(H2O2)', 'pH']
scaler3.fit(data1[feature_columns3]) 

min_max = {
    'pH': (data1['pH'].min(), data1['pH'].max()),
    'H2O2': (data1['lg(H2O2)'].min(), data1['lg(H2O2)'].max()),
    'O3': (data1['lg(O3)'].min(), data1['lg(O3)'].max())
}

feature_index_map = {0: 'O3', 1: 'H2O2', 2: 'pH'}

for cycle in range(10):
    print(f'cycle: {cycle+1}')

    valid_indices = [i for i in range(len(y_pred_orig)) 
                   if abs(df_GAN_origin.iloc[i]['TOC'] - y_pred_orig[i]) < 0.05]
    
    if not valid_indices:
        raise ValueError("No valid samples were found, please adjust the screening threshold.")

    try:
        selected_indices = np.random.choice(valid_indices, size=18, replace=False)
    except ValueError:
        selected_indices = np.random.choice(valid_indices, size=18, replace=True)
    
    combined_data = pd.concat([data1, df_GAN_origin.iloc[selected_indices]])
    X_final = scaler3.transform(combined_data[feature_columns3])
    y_final = combined_data['TOC'].values.reshape(-1, 1)
    
    best_params = run_bayesian_optimization(X_final, y_final)
    final_model = GradientBoostingRegressor(
        learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        min_samples_split=int(best_params['min_samples_split']),
        max_features=best_params['max_features'],
        max_depth=int(best_params['max_depth']),
        max_leaf_nodes=int(best_params['max_leaf_nodes']),
        random_state=2
    ).fit(X_final, y_final.ravel())
    
    def get_pdp(features):
        grid, _, pdp = partial_dependence_2d(
            final_model, X_final, features=features, grid_resolution=8
        )
        feature_names = [feature_index_map[f] for f in features]
        return [reverse_minmax_normalization(g, *min_max[col]) 
               for g, col in zip(grid, feature_names)], pdp

    (grid0, grid1), pdp1 = get_pdp([2, 1])  # pH and H2O2
    (grid02, grid12), pdp2 = get_pdp([2, 0])  # pH and O3
    (grid03, grid13), pdp3 = get_pdp([0, 1])  # O3 and H2O2
    
    all_pdp1.append(pdp1)
    all_pdp2.append(pdp2)
    all_pdp3.append(pdp3)
    grids.append((grid0, grid1, grid02, grid12, grid03, grid13))


with pd.ExcelWriter('pdp_results.xlsx') as writer:
    def save_sheet(name, grid_x, grid_y, pdp_list):
        df = pd.DataFrame()
        for i, (x, y) in enumerate(zip(grid_x.ravel(), grid_y.ravel())):
            df.loc[i, 'X'] = x
            df.loc[i, 'Y'] = y
            for c in range(10):
                df.loc[i, f'Cycle_{c+1}'] = pdp_list[c].ravel()[i]
            df.loc[i, 'Average'] = np.mean([pdp_list[c].ravel()[i] for c in range(10)])
        df.to_excel(writer, sheet_name=name, index=False)
    
    save_sheet('pH-H2O2', grids[0][0], grids[0][1], all_pdp1)
    save_sheet('pH-O3', grids[0][2], grids[0][3], all_pdp2)
    save_sheet('O3-H2O2', grids[0][4], grids[0][5], all_pdp3)

