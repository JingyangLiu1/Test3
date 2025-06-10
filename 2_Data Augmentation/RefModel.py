#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from bayes_opt import BayesianOptimization
import warnings

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


Aug_all = pd.read_excel(r"C:\Users\HP\Desktop\NKU\O3-based AOPs\Peroxne.xlsx",
                      sheet_name='RefData',
                      index_col=0,
                      header=0)
Aug = Aug_all[["lg(O3/DOC)", "lg(H2O2/O3)", "pH", "TOC removal(2h)"]]
Ref = Aug.iloc[0:36, :]  

data1 = Aug.iloc[36:54, :]  
features = ["lg(O3/DOC)", "lg(H2O2/O3)", "pH"]
target = "TOC removal(2h)"

r2_results = []
rmse_results = []

all_train_true = []
all_train_pred = []
all_test_true = []
all_test_pred = []

for iter in range(10):
    print(f"\n========== Iteration times {iter+1}/10 ==========")
    
    test_indices = np.random.choice(data1.index, size=4, replace=False)
    test_set = data1.loc[test_indices]
    train_set = data1.drop(test_indices)
    
    combined_data = pd.concat([train_set, Ref], axis=0)
    
    scaler = MinMaxScaler()
    
    X_combined = scaler.fit_transform(combined_data[features])
    y_combined = combined_data[target].values.reshape(-1, 1)
    
    X_augmented = X_combined  
    y_augmented = y_combined
    
    X_test = scaler.transform(test_set[features])
    y_test = test_set[target].values
    
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
    
    train_pred = final_model.predict(X_augmented)
    all_train_true.extend(y_augmented.ravel().tolist())
    all_train_pred.extend(train_pred.tolist())
    
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

