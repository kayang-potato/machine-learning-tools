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

class ModelParam():
    ''' 模型参数 '''

    def __init__(self):
        self.default_params = MODEL_PARAMS
        self.default_search_params = MODEL_GRID_SEARCH_PARAMS

    def show_models(self):
        ''' show model name '''
        return self.default_params.keys()

    def get_params(self, model_name, train_params=None):
        ''' 获取模型参数 '''
        model_params = self.default_params.get(model_name, {})
        if train_params:
            model_params.update(train_params)
        return model_params

    def get_search_params(self, model_name):
        ''' 获取模型搜索参数 '''
        return self.default_search_params.get(model_name, {})
