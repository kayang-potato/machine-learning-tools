'''
Author: kayang, kayang.name@outlook.com
Date: 2022-06-29 11:17:08
LastEditors: kayang
LastEditTime: 2022-07-15 12:01:33
Description: file content
Copyright (c) 2022 by kayang, All Rights Reserved.
'''

class ModelPredict():
    ''' 模型预测 '''
    def __init__(self, model):
        self.model = model
        self.model_name = self.get_name_from_model(model)

    def predict(self, test_x):
        if self.model_name in ['Lasso', 'Ridge', 'ElasticNet']:
            self.model.predict(test_x)
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
        return self.model

    def get_name_from_model(self, model):
        ''' 获取模型名称 '''
        if isinstance(model, LogisticRegression):
            return 'LogisticRegression'
        if isinstance(model, RandomForestClassifier):
            return 'RandomForestClassifier'
        if isinstance(model, RidgeClassifier):
            return 'RidgeClassifier'
        if isinstance(model, GradientBoostingClassifier):
            return 'GradientBoostingClassifier'
        if isinstance(model, ExtraTreesClassifier):
            return 'ExtraTreesClassifier'
        if isinstance(model, AdaBoostClassifier):
            return 'AdaBoostClassifier'
        if isinstance(model, SGDClassifier):
            return 'SGDClassifier'
        if isinstance(model, Perceptron):
            return 'Perceptron'
        if isinstance(model, PassiveAggressiveClassifier):
            return 'PassiveAggressiveClassifier'
        if isinstance(model, LinearRegression):
            return 'LinearRegression'
        if isinstance(model, RandomForestRegressor):
            return 'RandomForestRegressor'
        if isinstance(model, Ridge):
            return 'Ridge'
        if isinstance(model, ExtraTreesRegressor):
            return 'ExtraTreesRegressor'
        if isinstance(model, AdaBoostRegressor):
            return 'AdaBoostRegressor'
        if isinstance(model, RANSACRegressor):
            return 'RANSACRegressor'
        if isinstance(model, GradientBoostingRegressor):
            return 'GradientBoostingRegressor'
        if isinstance(model, Lasso):
            return 'Lasso'
        if isinstance(model, ElasticNet):
            return 'ElasticNet'
        if isinstance(model, LassoLars):
            return 'LassoLars'
        if isinstance(model, OrthogonalMatchingPursuit):
            return 'OrthogonalMatchingPursuit'
        if isinstance(model, BayesianRidge):
            return 'BayesianRidge'
        if isinstance(model, ARDRegression):
            return 'ARDRegression'
        if isinstance(model, SGDRegressor):
            return 'SGDRegressor'
        if isinstance(model, PassiveAggressiveRegressor):
            return 'PassiveAggressiveRegressor'
        if isinstance(model, MiniBatchKMeans):
            return 'MiniBatchKMeans'
        if isinstance(model, LinearSVR):
            return 'LinearSVR'
        if isinstance(model, LinearSVC):
            return 'LinearSVC'

        if xgb_installed:
            if isinstance(model, XGBClassifier):
                return 'XGBClassifier'
            if isinstance(model, XGBRegressor):
                return 'XGBRegressor'

        if keras_imported:
            if isinstance(model, KerasRegressor):
                return 'DeepLearningRegressor'
            if isinstance(model, KerasClassifier):
                return 'DeepLearningClassifier'

        if lgb_installed:
            if isinstance(model, LGBMClassifier):
                return 'LGBMClassifier'
            if isinstance(model, LGBMRegressor):
                return 'LGBMRegressor'

        if catboost_installed:
            if isinstance(model, CatBoostClassifier):
                return 'CatBoostClassifier'
            if isinstance(model, CatBoostRegressor):
                return 'CatBoostRegressor'

