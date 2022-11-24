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
from ml import ML


def load_data():
    ''' 加载测试数据 '''
    data = fetch_california_housing()  # 获取样例数据，这里的数据是加利福利亚的放假数据
    train_x = pd.DataFrame(data.data, columns=data.feature_names)
    train_y = data.target
    return train_x, train_y


def test_lasso(handle, x_train, x_test, y_train, y_test):
    ''' test lasso '''
    handle.init_model('lasso')
    handle.train(x_train, y_train)
    y_pred = handle.predict(x_test)
    print(f'Lasso MSE: {(mean_squared_error(y_test, y_pred))}')


def test_ridge(handle, x_train, x_test, y_train, y_test):
    ''' test ridge '''
    handle.init_model('ridge')
    handle.train(x_train, y_train)
    y_pred = handle.predict(x_test)
    print(f'Ridge MSE: {(mean_squared_error(y_test, y_pred))}')


def test_elastic(handle, x_train, x_test, y_train, y_test):
    ''' test elastic '''
    handle.init_model('elastic')
    handle.train(x_train, y_train)
    y_pred = handle.predict(x_test)
    print(f'Elastic MSE: {(mean_squared_error(y_test, y_pred))}')


if __name__ == "__main__":

    # 句柄
    handle_m = ML()

    # 加载数据
    data_x, data_y = load_data()
    train_x, test_x, train_y, test_y = train_test_split(data_x,
                                                        data_y,
                                                        test_size=0.3,
                                                        random_state=0)
    # lasso
    test_lasso(handle_m, train_x, test_x, train_y, test_y)

    # ridge
    test_ridge(handle_m, train_x, test_x, train_y, test_y)

    # elastic
    test_elastic(handle_m, train_x, test_x, train_y, test_y)
