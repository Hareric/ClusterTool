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
from sklearn.datasets import make_blobs
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.preprocessing import StandardScaler
import time


def snn_sim_matrix(X, k=5, t=1):
    """
    利用sklearn包中的KDTree,计算节点的共享最近邻相似度(SNN)矩阵
    :param X: array-like, shape = [samples_size, features_size]
    :param k: positive integer(default = 5), 计算snn相似度时从k个最近邻中确定
    :param t: positive integer(default = 1), 稀疏化相似度矩阵的阈值,只保留相似度不小于t的对象
    :return: 三元组表保存的相似度矩阵
    """
    try:
        X = np.array(X)
    except:
        raise ValueError("输入的数据集必须为矩阵")
    samples_size, features_size = X.shape  # 数据集样本的个数和特征的维数
    # knn_matrix = np.empty((samples_size, k), dtype=int)  # 记录每个样本的k个最近邻对应的索引
    # tree = KDTree(X)
    # for i in range(samples_size):
    #     ind = tree.query(X[i], k=k, return_distance=False)
    #     knn_matrix[i] = ind
    # print knn_matrix

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    knn_matrix = nbrs.kneighbors(X, return_distance=False)  # 记录每个样本的k个最近邻对应的索引
    snn_sim_table = []  # 三元组表记录相似度矩阵, .[0]与.[1]的相似度为.[2]
    for i in range(samples_size):
        for j in range(samples_size)[i + 1:]:
            sim_value = np.intersect1d(knn_matrix[i], knn_matrix[j]).__len__()  # 计算样本1与样本2的共享近邻点数
            if sim_value >= t:
                snn_sim_table.append([i, j, sim_value])
    return np.array(snn_sim_table)


if __name__ == '__main__':
    # X = [[1, 2, 3], [2, 4, 5], [2, 3, 1], [4, 5, 1]]
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    X = StandardScaler().fit_transform(X)
    print X
    print X.shape
    t1 = time.time()
    snn_sim_matrix(X, k=2)
    t2 = time.time()
    print t2 - t1