# coding=utf-8
#        ┏┓　　　┏┓+ +
# 　　　┏┛┻━━━┛┻┓ + +
# 　　　┃　　　　　　 ┃ 　
# 　　　┃　　　━　　　┃ ++ + + +
# 　　 ████━████ ┃+
# 　　　┃　　　　　　 ┃ +
# 　　　┃　　　┻　　　┃
# 　　　┃　　　　　　 ┃ + +
# 　　　┗━┓　　　┏━┛
# 　　　　　┃　　　┃　　　　　　　　　　　
# 　　　　　┃　　　┃ + + + +
# 　　　　　┃　　　┃　　　　Codes are far away from bugs with the animal protecting　　　
# 　　　　　┃　　　┃ + 　　　　神兽保佑,代码无bug　　
# 　　　　　┃　　　┃
# 　　　　　┃　　　┃　　+　　　　　　　　　
# 　　　　　┃　 　　┗━━━┓ + +
# 　　　　　┃ 　　　　　　　┣┓
# 　　　　　┃ 　　　　　　　┏┛
# 　　　　　┗┓┓┏━┳┓┏┛ + + + +
# 　　　　　　┃┫┫　┃┫┫
# 　　　　　　┗┻┛　┗┻┛+ + + +
"""
Author = Eric_Chan
Create_Time = 2016/08/05
"""
import numpy as np
from sklearn.neighbors import KDTree


def snn_sim_matrix(X, k=1):
    """
    利用sklearn包中的KDTree,计算节点的共享最近邻相似度(SNN)矩阵
    :param X: array-like, shape = [samples_size, features_size]
    :param k: positive integer(default = 1), SNN算法的阈值,只保留相似度不小于k的对象
    :return: 三元组表保存的稀疏矩阵
    """
    try:
        X = np.array(X)
    except ValueError:
        assert isinstance(X, list)

    # tree = KDTree(X)
    # three_tuple = np.array

if __name__ == '__main__':
    X = np.array([1,2,[3]])
    snn_sim_matrix(X)
