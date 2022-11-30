'''
Author: kayang, kayang.name@outlook.com
Date: 2022-06-29 11:17:08
LastEditors: kayang
LastEditTime: 2022-07-15 12:01:33
Description: file content
Copyright (c) 2022 by kayang, All Rights Reserved.
'''
from ml_header import *
from model_config import *

class ModelTrain():
    ''' 模型训练 '''

    def __init__(self, model_name=None, train_params=None):
        self.default_model_map = MODEL_FROM_NAME
        self.model_name = model_name
        self.train_params = train_params
        self.model = self.set_model(model_name, train_params)

    def set_model(self, model_name, train_params=None):
        ''' 设置模型类型以及参数 '''
        if not model_name:
            return None
        assert isinstance(model_name, str)
        assert isinstance(train_params, dict)
        try:
            model_without_params = self.default_model_map[model_name]
        except KeyError as error:
            raise error
        model_with_params = model_without_params.set_params(**train_params)
        self.model_name = model_name
        self.model = model_with_params

    def fit(self, train_x, train_y=None):
        ''' 训练模型 '''
        # TODO(kayang): 需要考虑不同模型训练方法
        if self.model_name in ['Lasso', 'Ridge', 'ElasticNet', 'LinearRegression', 'SVR']:
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
