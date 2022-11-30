'''
Author: kayang, kayang.name@outlook.com
Date: 2022-06-29 11:17:08
LastEditors: kayang
LastEditTime: 2022-07-15 12:01:33
Description: file content
Copyright (c) 2022 by kayang, All Rights Reserved.
'''
# pylint: disable=W0611
# pylint: disable=W0401
# pylint: disable=W0614
import scipy
import pandas as pd

# RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor,
# AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier,
# ExtraTreesClassifier, AdaBoostClassifier
from sklearn.ensemble import *

# RANSACRegressor, LinearRegression, Ridge, Lasso, ElasticNet, LassoLars,
# OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor,
# PassiveAggressiveRegressor, LogisticRegression, RidgeClassifier,
# SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.linear_model import *
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.cluster import MiniBatchKMeans

XGB_INSTALLED = False
try:
    # pylint: disable=E0401
    from xgboost import XGBClassifier, XGBRegressor
    XGB_INSTALLED = True
except ImportError:
    pass
    # print("xgboost not installed")

LGB_INSTALLED = False
try:
    # pylint: disable=E0401
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGB_INSTALLED = True
except ImportError:
    pass
    # print("lightgbm not installed")

CATBOOST_INSTALLED = False
try:
    # pylint: disable=E0401
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_INSTALLED = True
except ImportError:
    pass
    # print("catboost not installed")
