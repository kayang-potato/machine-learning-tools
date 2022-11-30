'''
Author: kayang, kayang.name@outlook.com
Date: 2022-06-29 11:17:08
LastEditors: kayang
LastEditTime: 2022-07-15 12:01:33
Description: file content
Copyright (c) 2022 by kayang, All Rights Reserved.
'''
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import pandas as pd
from model_params import ModelParam
from model_train import ModelTrain
from model_predict import ModelPredict

mpa = ModelParam()
mt = ModelTrain()
mpr = ModelPredict()

def load_data():
    ''' 加载测试数据 '''
    data = fetch_california_housing()  # 获取样例数据，这里的数据是加利福利亚的放假数据
    # pylint: disable=W0621
    # pylint: disable=E1101
    train_x = pd.DataFrame(data.data, columns=data.feature_names)
    train_y = data.target
    return train_x, train_y

def train_predict(model_name, x_train, x_test, y_train):
    model_params = mpa.get_params(model_name=model_name)
    mt.set_model(model_name, model_params)
    model = mt.fit(x_train, y_train)
    mpr.set_model(model)
    y_pred = mpr.predict(x_test)
    return y_pred

def test_lasso(x_train, x_test, y_train, y_test):
    ''' test lasso '''
    model_name = 'Lasso'
    y_pred = train_predict(model_name, x_train, x_test, y_train)
    print(f'Lasso MSE: {(mean_squared_error(y_test, y_pred))}')


def test_ridge(x_train, x_test, y_train, y_test):
    ''' test ridge '''
    model_name = 'Ridge'
    y_pred = train_predict(model_name, x_train, x_test, y_train)
    print(f'Ridge MSE: {(mean_squared_error(y_test, y_pred))}')


def test_elastic(x_train, x_test, y_train, y_test):
    ''' test elastic '''
    model_name = 'ElasticNet'
    y_pred = train_predict(model_name, x_train, x_test, y_train)
    print(f'Elastic MSE: {(mean_squared_error(y_test, y_pred))}')

def test_linear_regression(x_train, x_test, y_train, y_test):
    ''' test linear_regression '''
    model_name = 'LinearRegression'
    y_pred = train_predict(model_name, x_train, x_test, y_train)
    print(f'LinearRegression MSE: {(mean_squared_error(y_test, y_pred))}')

def test_svr(x_train, x_test, y_train, y_test):
    ''' test svr '''
    model_name = 'SVR'
    y_pred = train_predict(model_name, x_train, x_test, y_train)
    print(f'SVR MSE: {(mean_squared_error(y_test, y_pred))}')


if __name__ == "__main__":

    # 加载数据
    data_x, data_y = load_data()
    train_x, test_x, train_y, test_y = train_test_split(data_x,
                                                        data_y,
                                                        test_size=0.3,
                                                        random_state=0)
    # lasso
    test_lasso(train_x, test_x, train_y, test_y)

    # ridge
    test_ridge(train_x, test_x, train_y, test_y)

    # elastic
    test_elastic(train_x, test_x, train_y, test_y)
    
    # linear_regression
    test_linear_regression(train_x, test_x, train_y, test_y)

    # SVR
    test_svr(train_x, test_x, train_y, test_y)

