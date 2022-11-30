'''
Author: kayang, kayang.name@outlook.com
Date: 2022-06-29 11:17:08
LastEditors: kayang
LastEditTime: 2022-07-15 12:01:33
Description: file content
Copyright (c) 2022 by kayang, All Rights Reserved.
'''

from ml_header import *

# 模型参数初始化配置
MODEL_PARAMS = {
    'Lasso': {
        'alpha': 0.05
    },
    'Ridge': {
        'alpha': 0.05
    },
    'ElasticNet': {
        'alpha': 0.05,
        'l1_ratio': 0.5
    },
    'LinearRegression': {
        'fit_intercept': True,
        'copy_X': True,
        'n_jobs': -2
    },
    'LinearSVR': {
        'C': 1.0
    },
    'SVR': {
        'gamma': 0.0004,
        'kernel': 'rbf',
        'C': 15,
        'epsilon': 0.009
    },
    'LogisticRegression': {},
    'RandomForestClassifier': {
        'n_jobs': -2,
        'n_estimators': 30
    },
    'ExtraTreesClassifier': {
        'n_jobs': -1
    },
    'AdaBoostClassifier': {},
    'SGDClassifier': {
        'n_jobs': -1
    },
    'Perceptron': {
        'n_jobs': -1
    },
    'RandomForestRegressor': {
        'n_jobs': -2,
        'n_estimators': 30
    },
    'LinearSVR': {
        'dual': False,
        'loss': 'squared_epsilon_insensitive'
    },
    'ExtraTreesRegressor': {
        'n_jobs': -1
    },
    'MiniBatchKMeans': {
        'n_clusters': 8
    },
    'GradientBoostingRegressor': {
        'presort': False,
        'learning_rate': 0.1,
        'warm_start': True
    },
    'GradientBoostingClassifier': {
        'presort': False,
        'learning_rate': 0.1,
        'warm_start': True
    },
    'SGDRegressor': {
        'shuffle': False
    },
    'PassiveAggressiveRegressor': {
        'shuffle': False
    },
    'AdaBoostRegressor': {},
    'LGBMRegressor': {
        'n_estimators': 500,
        'learning_rate': 0.15,
        'num_leaves': 8,
        'lambda_l2': 0.001,
        'histogram_pool_size': 16384
    },
    'LGBMClassifier': {
        'n_estimators': 500,
        'learning_rate': 0.15,
        'num_leaves': 8,
        'lambda_l2': 0.001,
        'histogram_pool_size': 16384
    },
    'CatBoostRegressor': {},
    'CatBoostClassifier': {}
}

# 网格搜索初始化参数
MODEL_GRID_SEARCH_PARAMS = {
    'DeepLearningRegressor': {
        'hidden_layers': [[1], [1, 0.1], [1, 1, 1], [1, 0.5, 0.1], [2], [5],
                          [1, 0.5, 0.25, 0.1, 0.05], [1, 1, 1, 1], [1, 1]],
        'dropout_rate': [0.0, 0.2, 0.4, 0.6, 0.8],
        'kernel_initializer': [
            'uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
            'glorot_uniform', 'he_normal', 'he_uniform'
        ],
        'activation': [
            'tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu',
            'sigmoid', 'hard_sigmoid', 'linear', 'LeakyReLU', 'PReLU', 'ELU',
            'ThresholdedReLU'
        ],
        'batch_size': [16, 32, 64, 128, 256, 512],
        'optimizer':
        ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    },
    'DeepLearningClassifier': {
        'hidden_layers': [[1], [0.5], [2], [1, 1], [0.5, 0.5],
                          [2, 2], [1, 1, 1], [1, 0.5, 0.5], [0.5, 1, 1],
                          [1, 0.5, 0.25], [1, 2, 1], [1, 1, 1, 1],
                          [1, 0.66, 0.33, 0.1], [1, 2, 2, 1]],
        'batch_size': [16, 32, 64, 128, 256, 512],
        'optimizer':
        ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
        'activation': [
            'tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu',
            'sigmoid', 'hard_sigmoid', 'linear', 'LeakyReLU', 'PReLU', 'ELU',
            'ThresholdedReLU'
        ]
        # , 'epochs': [2, 4, 6, 10, 20]
        # , 'batch_size': [10, 25, 50, 100, 200, 1000]
        # , 'lr': [0.001, 0.01, 0.1, 0.3]
        # , 'momentum': [0.0, 0.3, 0.6, 0.8, 0.9]
        # , 'init_mode': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
        # , 'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        # , 'weight_constraint': [1, 3, 5]
        ,
        'dropout_rate': [0.0, 0.3, 0.6, 0.8, 0.9]
    },
    'XGBClassifier': {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 75, 100, 150, 200, 375, 500, 750, 1000],
        'min_child_weight': [1, 5, 10, 50],
        'subsample': [0.5, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0]
        # 'subsample': [0.5, 1.0]
        # 'lambda': [0.9, 1.0]
    },
    'XGBRegressor': {
        # Add in max_delta_step if classes are extremely imbalanced
        'max_depth': [1, 3, 8, 25],
        # 'lossl': ['ls', 'lad', 'huber', 'quantile'],
        # 'booster': ['gbtree', 'gblinear', 'dart'],
        # 'objective': ['reg:linear', 'reg:gamma'],
        # 'learning_rate': [0.01, 0.1],
        'subsample': [0.5, 1.0]
        # 'subsample': [0.4, 0.5, 0.58, 0.63, 0.68, 0.76],
    },
    'GradientBoostingRegressor': {
        # Add in max_delta_step if classes are extremely imbalanced
        'max_depth': [1, 2, 3, 4, 5, 7, 10, 15],
        'max_features': ['sqrt', 'log2', None],
        'loss': ['ls', 'huber'],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'n_estimators': [10, 50, 75, 100, 125, 150, 200, 500, 1000, 2000],
        'subsample': [0.5, 0.65, 0.8, 0.9, 0.95, 1.0]
    },
    'GradientBoostingClassifier': {
        'loss': ['deviance', 'exponential'],
        'max_depth': [1, 2, 3, 4, 5, 7, 10, 15],
        'max_features': ['sqrt', 'log2', None],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
        'n_estimators': [10, 50, 75, 100, 125, 150, 200, 500, 1000, 2000],
    },
    'LogisticRegression': {
        'C': [.0001, .001, .01, .1, 1, 10, 100, 1000],
        'class_weight': [None, 'balanced'],
        'solver': ['newton-cg', 'lbfgs', 'sag']
    },
    'LinearRegression': {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    },
    'RandomForestClassifier': {
        'criterion': ['entropy', 'gini'],
        'class_weight': [None, 'balanced'],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 20, 50, 100],
        'min_samples_leaf': [1, 2, 5, 20, 50, 100],
        'bootstrap': [True, False]
    },
    'RandomForestRegressor': {
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_samples_split': [2, 5, 20, 50, 100],
        'min_samples_leaf': [1, 2, 5, 20, 50, 100],
        'bootstrap': [True, False]
    },
    'RidgeClassifier': {
        'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
        'class_weight': [None, 'balanced'],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
    },
    'Ridge': {
        'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
    },
    'ExtraTreesRegressor': {
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_samples_split': [2, 5, 20, 50, 100],
        'min_samples_leaf': [1, 2, 5, 20, 50, 100],
        'bootstrap': [True, False]
    },
    'AdaBoostRegressor': {
        'base_estimator': [None, LinearRegression(n_jobs=-1)],
        'loss': ['linear', 'square', 'exponential']
    },
    'RANSACRegressor': {
        'min_samples': [None, .1, 100, 1000, 10000],
        'stop_probability': [0.99, 0.98, 0.95, 0.90]
    },
    'Lasso': {
        'selection': ['cyclic', 'random'],
        'tol': [.0000001, .000001, .00001, .0001, .001],
        'positive': [True, False]
    },
    'ElasticNet': {
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'selection': ['cyclic', 'random'],
        'tol': [.0000001, .000001, .00001, .0001, .001],
        'positive': [True, False]
    },
    'LassoLars': {
        'positive': [True, False],
        'max_iter': [50, 100, 250, 500, 1000]
    },
    'OrthogonalMatchingPursuit': {
        'n_nonzero_coefs': [None, 3, 5, 10, 25, 50, 75, 100, 200, 500]
    },
    'BayesianRidge': {
        'tol': [.0000001, .000001, .00001, .0001, .001],
        'alpha_1': [.0000001, .000001, .00001, .0001, .001],
        'lambda_1': [.0000001, .000001, .00001, .0001, .001],
        'lambda_2': [.0000001, .000001, .00001, .0001, .001]
    },
    'ARDRegression': {
        'tol': [.0000001, .000001, .00001, .0001, .001],
        'alpha_1': [.0000001, .000001, .00001, .0001, .001],
        'alpha_2': [.0000001, .000001, .00001, .0001, .001],
        'lambda_1': [.0000001, .000001, .00001, .0001, .001],
        'lambda_2': [.0000001, .000001, .00001, .0001, .001],
        'threshold_lambda': [100, 1000, 10000, 100000, 1000000]
    },
    'SGDRegressor': {
        'loss': [
            'squared_loss', 'huber', 'epsilon_insensitive',
            'squared_epsilon_insensitive'
        ],
        'penalty': ['none', 'l2', 'l1', 'elasticnet'],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'alpha': [.0000001, .000001, .00001, .0001, .001]
    },
    'PassiveAggressiveRegressor': {
        'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        'C': [.0001, .001, .01, .1, 1, 10, 100, 1000],
    },
    'SGDClassifier': {
        'loss': [
            'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
            'squared_loss', 'huber', 'epsilon_insensitive',
            'squared_epsilon_insensitive'
        ],
        'penalty': ['none', 'l2', 'l1', 'elasticnet'],
        'alpha': [.0000001, .000001, .00001, .0001, .001],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'class_weight': ['balanced', None]
    },
    'Perceptron': {
        'penalty': ['none', 'l2', 'l1', 'elasticnet'],
        'alpha': [.0000001, .000001, .00001, .0001, .001],
        'class_weight': ['balanced', None]
    },
    'PassiveAggressiveClassifier': {
        'loss': ['hinge', 'squared_hinge'],
        'class_weight': ['balanced', None],
        'C': [0.01, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    },
    'LGBMClassifier': {
        'boosting_type': ['gbdt', 'dart'],
        'min_child_samples':
        [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000],
        'num_leaves': [
            2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150,
            200, 250
        ],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'subsample': [0.7, 0.9, 1.0],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators':
        [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000]
    },
    'LGBMRegressor': {
        'boosting_type': ['gbdt', 'dart'],
        'min_child_samples':
        [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000],
        'num_leaves': [
            2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150,
            200, 250
        ],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'subsample': [0.7, 0.9, 1.0],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators':
        [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000]
    },
    'CatBoostClassifier': {
        'depth': [1, 2, 3, 5, 7, 9, 12, 15, 20, 32],
        'l2_leaf_reg': [.0000001, .000001, .00001, .0001, .001, .01, .1],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    },
    'CatBoostRegressor': {
        'depth': [1, 2, 3, 5, 7, 9, 12, 15, 20, 32],
        'l2_leaf_reg': [.0000001, .000001, .00001, .0001, .001, .01, .1],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    },
    'LinearSVR': {
        'C': [0.5, 0.75, 0.85, 0.95, 1.0],
        'epsilon': [0, 0.05, 0.1, 0.15, 0.2]
    },
    'LinearSVC': {
        'C': [0.5, 0.75, 0.85, 0.95, 1.0]
    }
}

# 模型名映射
MODEL_FROM_NAME = {
    # Classifiers
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'RidgeClassifier': RidgeClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'LinearSVC': LinearSVC(),

    # Regressors
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'LinearRegression': LinearRegression(),
    'SVR': SVR(),
    'RandomForestRegressor': RandomForestRegressor(),
    'LinearSVR': LinearSVR(),
    'ExtraTreesRegressor': ExtraTreesRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'RANSACRegressor': RANSACRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'ElasticNet': ElasticNet(),
    'LassoLars': LassoLars(),
    'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
    'BayesianRidge': BayesianRidge(),
    'ARDRegression': ARDRegression(),

    # Clustering
    'MiniBatchKMeans': MiniBatchKMeans(),

}
if XGB_INSTALLED:
    MODEL_FROM_NAME['XGBClassifier'] = XGBClassifier()
    MODEL_FROM_NAME['XGBRegressor'] = XGBRegressor()

if LGB_INSTALLED:
    MODEL_FROM_NAME['LGBMRegressor'] = LGBMRegressor()
    MODEL_FROM_NAME['LGBMClassifier'] = LGBMClassifier()

if CATBOOST_INSTALLED:
    MODEL_FROM_NAME['CatBoostRegressor'] = CatBoostRegressor(
        calc_feature_importance=True)
    MODEL_FROM_NAME['CatBoostClassifier'] = CatBoostClassifier(
        calc_feature_importance=True)

# 模型映射至模型名称
def NAME_FROM_MODEL(model):
    ''' 获取模型名称 '''
    if isinstance(model, LogisticRegression):
        return 'LogisticRegression'
    if isinstance(model, SVR):
        return 'SVR'
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

    if XGB_INSTALLED:
        if isinstance(model, XGBClassifier):
            return 'XGBClassifier'
        if isinstance(model, XGBRegressor):
            return 'XGBRegressor'

    if LGB_INSTALLED:
        if isinstance(model, LGBMClassifier):
            return 'LGBMClassifier'
        if isinstance(model, LGBMRegressor):
            return 'LGBMRegressor'

    if CATBOOST_INSTALLED:
        if isinstance(model, CatBoostClassifier):
            return 'CatBoostClassifier'
        if isinstance(model, CatBoostRegressor):
            return 'CatBoostRegressor'
    else:
        return None
