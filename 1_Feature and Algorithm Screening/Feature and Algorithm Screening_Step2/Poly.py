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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import RFE
from bayes_opt import BayesianOptimization

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor

FEATURE_COMBINATIONS = [
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC', 'H2O2_60', 'FMax4'],
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC', 'Φn1', 'FMax1'],
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC', 'FRI', 'FMax2'],
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC', 'Φn1', 'FMax3'],
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC', 'FMax2', 'FMax4'],
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC', 'O3(1)120'],
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC', 'OU120'],
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC', 'H2O2_60'],
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC', 'Φn4'],
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC', 'H2O2_120'],
    ['lg(O3)', 'lg(H2O2)', 'pH','TOC']
]

data = pd.read_excel(r"C:\Users\HP\Desktop\Data.xlsx", 
                    sheet_name='16+3', 
                    index_col=0, 
                    header=0)
raw_data = data.iloc[0:18, :]  


def evaluate_features(features, n_runs=10):

    OP = raw_data[features]
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(OP)
    data_processed = pd.DataFrame(scaled_data, columns=features)
    
    X = data_processed[[col for col in features if col != 'TOC']]
    y = data_processed['TOC']
    
    results = {'R2': [], 'RMSE': []}
    
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=4, random_state=run
        )
        
        poly=PolynomialFeatures(degree=2,interaction_only=True)
        Xtrain_poly = poly.fit_transform(X_train)
        model = LinearRegression()
        model.fit(Xtrain_poly, y_train.ravel())
        Xtest_poly = poly.fit_transform(X_test)
        y_pred = model.predict(Xtest_poly)
        
        results['R2'].append(r2_score(y_test, y_pred))
        results['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    return {
        'mean_R2': np.mean(results['R2']),
        'std_R2': np.std(results['R2']),
        'mean_RMSE': np.mean(results['RMSE']),
        'std_RMSE': np.std(results['RMSE'])
    }

if __name__ == "__main__":
    final_results = {}
    
    for feature_set in FEATURE_COMBINATIONS:
        print(f"\n{'='*40}")
        print(f"The feature combination is being evaluated: {feature_set}")
        try:
            metrics = evaluate_features(feature_set)
            final_results[str(feature_set)] = metrics
            print(f"Evaluation completed: R² = {metrics['mean_R2']:.4f} ± {metrics['std_R2']:.4f}, "
                  f"RMSE = {metrics['mean_RMSE']:.4f} ± {metrics['std_RMSE']:.4f}")
        except KeyError as e:
            print(f"Error : The feature column { e } is missing from the data, please check the name of the feature")
            final_results[str(feature_set)] = "Assessment failed"
    
    print("\n\n=== Final evaluation results ===")
    for feature_set, metrics in final_results.items():
        if isinstance(metrics, dict):
            print(f"\n Feature combination: {feature_set}")
            print(f"AverageR²: {metrics['mean_R2']:.4f} ± {metrics['std_R2']:.4f}")
            print(f"AverageRMSE: {metrics['mean_RMSE']:.4f} ± {metrics['std_RMSE']:.4f}")
        else:
            print(f"\n Feature combination: {feature_set} -> {metrics}")

