'''
Author: kayang, kayang.name@outlook.com
Date: 2022-06-29 11:17:08
LastEditors: kayang
LastEditTime: 2022-07-15 12:01:33
Description: file content
Copyright (c) 2022 by kayang, All Rights Reserved.
'''

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


class ML():
    ''' 机器学习 '''

    def __init__(self):
        self.model_name = None
        self.handle = None

    def init_model(self, model_name):
        ''' init model '''
        self.model_name = model_name
        if model_name == 'lasso':
            model = self.lasso()
        elif model_name == 'ridge':
            model = self.ridge()
        elif model_name == 'elastic':
            model = self.elastic()
        self.handle = model

    def train(self, x_train, y_train):
        ''' train '''
        self.handle.fit(x_train, y_train)

    def predict(self, x_test):
        ''' predict '''
        y_pred = self.handle.predict(x_test)
        return y_pred

    def lasso(self):
        ''' lasso '''
        cur_model = Lasso(alpha=0.05)
        return cur_model

    def ridge(self):
        ''' ridge '''
        cur_model = Ridge(alpha=0.05)
        return cur_model

    def elastic(self):
        ''' elastic '''
        cur_model = ElasticNet(alpha=0.05,l1_ratio=0.5)
        return cur_model
