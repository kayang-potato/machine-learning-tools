'''
Author: kayang, kayang.name@outlook.com
Date: 2022-06-29 11:17:08
LastEditors: kayang
LastEditTime: 2022-07-15 12:01:33
Description: file content
Copyright (c) 2022 by kayang, All Rights Reserved.
'''
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
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.cluster import MiniBatchKMeans

XGB_INSTALLED = False
try:
    # pylint: disable=E0401
    from xgboost import XGBClassifier, XGBRegressor
    XGB_INSTALLED = True
except ImportError:
    print("xgboost not installed")

LGB_INSTALLED = False
try:
    # pylint: disable=E0401
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGB_INSTALLED = True
except ImportError:
    print("lightgbm not installed")

CATBOOST_INSTALLED = False
try:
    # pylint: disable=E0401
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_INSTALLED = True
except ImportError:
    print("catboost not installed")


class ModelTrain():
    ''' 模型训练 '''

    def __init__(self, model_name, train_params):
        self.default_model_map = self._default_model_map()
        self.model_name = model_name
        self.model = self.set_model(model_name, train_params)

    def set_model(self, model_name, train_params=None):
        assert isinstance(model_name, str)
        assert isinstance(train_params, dict)
        try:
            model_without_params = self.default_model_map[model_name]
        except KeyError as error:
            raise error
        model_with_params = model_without_params.set_params(**train_params)
        return model_with_params

    def fit(self, train_x, train_y=None):
        ''' 训练模型 '''
        # TODO(kayang): 需要考虑不同模型训练方法
        if self.model_name in ['Lasso', 'Ridge', 'ElasticNet']:
            self.model.fit(train_x, train_y)
        elif self.model_name in [
                'BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit',
                'ARDRegression', 'Perceptron', 'PassiveAggressiveClassifier',
                'SGDClassifier', 'RidgeClassifier', 'LogisticRegression',
                'XGBClassifier', 'XGBRegressor'
        ]:
            pass
        elif self.model_name[:4] == 'LGBM':
            pass
        elif self.model_name[:8] == 'CatBoost':
            pass
        elif self.model_name[:16] == 'GradientBoosting':
            pass
        else:
            self.model.fit(train_x, train_y)
        return self.model

    def fit_grid_search(self, train_x, train_y=None, grid_search_params=None):
        ''' 网格搜索 '''
        # from sklearn.model_selection import GridSearchCV, KFold, train_test_split
        pass

    def _default_model_map(self):
        ''' 默认模型映射 '''
        model_map = {
            # Classifiers
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'RidgeClassifier': RidgeClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'LinearSVC': LinearSVC(),

            # Regressors
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'Ridge': Ridge(),
            'LinearSVR': LinearSVR(),
            'ExtraTreesRegressor': ExtraTreesRegressor(),
            'AdaBoostRegressor': AdaBoostRegressor(),
            'RANSACRegressor': RANSACRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'LassoLars': LassoLars(),
            'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
            'BayesianRidge': BayesianRidge(),
            'ARDRegression': ARDRegression(),

            # Clustering
            'MiniBatchKMeans': MiniBatchKMeans(),
        }
        if XGB_INSTALLED:
            model_map['XGBClassifier'] = XGBClassifier()
            model_map['XGBRegressor'] = XGBRegressor()

        if LGB_INSTALLED:
            model_map['LGBMRegressor'] = LGBMRegressor()
            model_map['LGBMClassifier'] = LGBMClassifier()

        if CATBOOST_INSTALLED:
            model_map['CatBoostRegressor'] = CatBoostRegressor(
                calc_feature_importance=True)
            model_map['CatBoostClassifier'] = CatBoostClassifier(
                calc_feature_importance=True)

        return model_map
