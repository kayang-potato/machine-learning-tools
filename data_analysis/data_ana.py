'''
Author: Yang Kaihang, kayang.name@outlook.com
Date: 2022-06-29 11:17:08
LastEditors: Yang Kaihang
LastEditTime: 2022-07-15 12:01:33
Description: file content
Copyright (c) 2022 by Yang Kaihang, All Rights Reserved.
'''

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataProcessor():
    ''' 数据处理 '''

    def __init__(self, data, root_dir="./", target=-1):
        assert isinstance(data, pd.core.frame.DataFrame)
        self.data = data
        self.root_dir = root_dir
        self.target_label = data.columns[target]
        self.top_cols = []

    def reset_data(self, data, root_dir="./", target=-1):
        ''' 重置panda数据 '''
        self.data = data
        self.root_dir = root_dir
        self.target_label = data.columns[target]

    def observe(self, cor_topk=10):
        '''
        观察数据维度、数据头（前10个数据及列名）、数据信息（非空数量和数据类型）、数据描述（均值方差中位数等）、
        特征之间的相关性（数值型特征的热力图）、与目标前topk相关的热力图
        '''
        print("observe start: ")
        pd.set_option('display.max_columns', None)
        print(f"#data shape: {self.data.shape}")
        print(f"#data head: \n{self.data.head()}")
        print("#data info: ")
        self.data.info()
        print("#data describe: \n{self.data.describe()}")
        print("correlation: ")
        corrmat = self.data.corr()
        plt.figure(figsize=(12, 12))
        plt.title('Correlation of Features', y=1.00, size=20)
        sns.heatmap(corrmat,
                    linewidths=0.1,
                    vmax=1,
                    square=True,
                    cmap=plt.cm.viridis,
                    linecolor='white',
                    annot=False)
        corr_path = os.path.join(self.root_dir, "data_corr.jpg")
        plt.savefig(self._check_filename_available(corr_path))
        plt.close()
        print(f"Correlation of Features path: {corr_path}")
        top_cols = corrmat.nlargest(cor_topk,
                                    self.target_label)[self.target_label]
        self.top_cols = top_cols.index
        corrmat_topk = self.data[top_cols.index].corr()
        plt.figure(figsize=(12, 12))
        plt.title('Correlation of topk_Features', y=1.00, size=20)
        sns.heatmap(corrmat_topk,
                    linewidths=0.1,
                    vmax=1,
                    square=True,
                    cmap=plt.cm.viridis,
                    linecolor='white',
                    annot=True)
        corr_topk_path = os.path.join(self.root_dir, "data_topk_corr.jpg")
        plt.savefig(self._check_filename_available(corr_topk_path))
        plt.close()
        print(f"target label: {self.target_label}")
        print(f"top correlation cols: \n{top_cols}")

    def plot(self, plt_x_size=16, plt_y_size=20, x_col=[], y_col=[]):
        # 分离数字特征和类别特征，画数值型特征与目标特征的散点图
        # 默认画所有数值型特征和目标特征的散点图，非默认下适应输入
        scatter_path = os.path.join(self.root_dir,
                                    "feature2target_scatters.jpg")
        plt.figure(figsize=(plt_x_size, plt_y_size))
        plt.subplots_adjust(hspace=1, wspace=0.3)
        column_size = 3
        if len(x_col) > 0:
            assert len(y_col) > 0
            feature_cnt = len(x_col) * len(y_col)
            row_size = math.ceil(feature_cnt / column_size)
            column_size = min(len(x_col), column_size)
            cur_scatter_pos = 1
            for _, x_feature in enumerate(x_col):
                for _, y_feature in enumerate(y_col):
                    if self.data[x_feature].dtype != 'object':
                        plt.subplot(row_size, column_size, cur_scatter_pos)
                        sns.scatterplot(x=x_feature,
                                        y=y_feature,
                                        data=self.data,
                                        alpha=0.5,
                                        color='b')
                    else:
                        plt.subplot(row_size, column_size, cur_scatter_pos)
                        sns.boxplot(x=x_feature, y=y_feature, data=self.data)
                        plt.xticks(rotation=90, fontsize=12)
                    cur_scatter_pos += 1
        else:
            num_features = []
            cate_features = []
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    cate_features.append(col)
                else:
                    num_features.append(col)
            print('number of numeric features:', len(num_features))
            print('number of categorical features:', len(cate_features))
            #查看数字特征与目标值的关系
            row_size = math.ceil(
                (len(num_features) + len(cate_features)) / column_size)
            column_size = min(
                len(num_features) + len(cate_features), column_size)
            cur_scatter_pos = 1
            for _, feature in enumerate(num_features):
                plt.subplot(row_size, column_size, cur_scatter_pos)
                sns.scatterplot(x=feature,
                                y=self.target_label,
                                data=self.data,
                                alpha=0.5,
                                color='b')
                cur_scatter_pos += 1
            for _, feature in enumerate(cate_features):
                plt.subplot(row_size, column_size, cur_scatter_pos)
                sns.boxplot(x=feature, y=self.target_label, data=self.data)
                plt.xticks(rotation=90, fontsize=12)
                cur_scatter_pos += 1
        plt.savefig(self._check_filename_available(scatter_path))
        plt.close()
        if len(self.top_cols) > 0:
            self._plot_topk(min(len(self.top_cols), 10))

    def _plot_topk(self, topk=10):
        ''' 与目标变量最大相关的前十散点图和分布图 '''
        g_sns = sns.PairGrid(self.data[self.top_cols[:topk]])
        g_sns.map_diag(plt.hist)
        g_sns.map_offdiag(plt.scatter)
        scatter_topk_path = os.path.join(self.root_dir,
                                         "topk2topk_scatter.jpg")
        plt.savefig(self._check_filename_available(scatter_topk_path))
        plt.close()

    def _check_filename_available(self, filename):
        ''' 循环判断文件名是否存在并返回新文件名 '''
        label = [0]
        def check_meta(file_name):
            file_name_new = file_name
            if os.path.isfile(file_name):
                file_name_new = file_name[:file_name.rfind('.')] + '_' + str(
                    label[0]) + file_name[file_name.rfind('.'):]
                label[0] += 1
            if os.path.isfile(file_name_new):
                file_name_new = check_meta(file_name)
            return file_name_new

        return_name = check_meta(filename)
        return return_name
