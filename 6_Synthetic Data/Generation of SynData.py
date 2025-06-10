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

np.random.seed(42)

data = np.random.rand(1000, 3)
A = data[:, 0]
B = data[:, 1]
C = data[:, 2]

a = (1.5 * A)**2

b = np.where(B < 0.3, -B, B - 0.6)

c = np.where(C < 0.5, (C + 1)**2, 2.75 - C)

noise = lambda: np.random.normal(0, 0.1, 1000)

T = a + b + c + noise()
D = 0.3*(a**2 + b**2) + noise()
E = 0.6*(a + c) + noise()
F = 0.8*(b + 1)**2 + noise()
G = a + 2*b + 3*c + noise()

df = pd.DataFrame({
    'A': A,
    'B': B,
    'C': C,
    'D': D,
    'E': E,
    'F': F,
    'G': G,
    'T': T})

