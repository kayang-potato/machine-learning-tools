'''
Author: kayang, kayang.name@outlook.com
Date: 2022-06-29 11:17:08
LastEditors: kayang
LastEditTime: 2022-07-15 12:01:33
Description: file content
Copyright (c) 2022 by kayang, All Rights Reserved.
'''

# pylint: disable=W0401
# pylint: disable=W0614
from ml_header import *
from model_config import *

class ModelPredict():
    ''' 模型预测 '''
    def __init__(self, model=None):
        self.model = model
        self.get_name_from_model = NAME_FROM_MODEL
        self.model_name = self.get_name_from_model(model)

    def set_model(self, model):
        self.model_name = self.get_name_from_model(model)
        self.model = model

    def predict(self, test_x):
        ''' 预测 '''
        assert self.model
        if self.model_name in ['Lasso', 'Ridge', 'ElasticNet', 'LinearRegression', 'SVR']:
            pred_y = self.model.predict(test_x)
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
            self.model.predict(test_x)
        return pred_y
