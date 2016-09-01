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
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
import time


def snn_sim_matrix(X, k=5):
    """
    利用sklearn包中的KDTree,计算节点的共享最近邻相似度(SNN)矩阵
    :param X: array-like, shape = [samples_size, features_size]
    :param k: positive integer(default = 5), 计算snn相似度的阈值k
    :return: snn距离矩阵
    """
    try:
        X = np.array(X)
    except:
        raise ValueError("输入的数据集必须为矩阵")
    samples_size, features_size = X.shape  # 数据集样本的个数和特征的维数
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    knn_matrix = nbrs.kneighbors(X, return_distance=False)  # 记录每个样本的k个最近邻对应的索引
    sim_matrix = 0.5 + np.zeros((samples_size, samples_size))  # snn相似度矩阵
    for i in range(samples_size):
        t = np.where(knn_matrix == i)[0]
        c = list(combinations(t, 2))
        for j in c:
            if j[0] not in knn_matrix[j[1]]:
                continue
            sim_matrix[j[0]][j[1]] += 1
    sim_matrix = 1 / sim_matrix  # 将相似度矩阵转化为距离矩阵
    sim_matrix = np.triu(sim_matrix)
    sim_matrix += sim_matrix.T - np.diag(sim_matrix.diagonal())
    return sim_matrix

if __name__ == '__main__':
    from sklearn.cluster import DBSCAN
    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib.pyplot as plt

    # 构建数据集
    # centers = [37, 4]
    # centers_2 = [-37, 4]
    # X, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=20)
    # X_2, l_2 = make_blobs(n_samples=50, cluster_std=8, centers=centers_2)
    # X = np.concatenate((X, X_2))
    # np.save('other/dataSet/diff_density_data', X)
    X = np.load('other/dataSet/diff_density_data.npy')

    # 基于snn相似度的聚类
    t1 = time.time()
    sim_matrix = snn_sim_matrix(X, k=8)
    t2 = time.time()
    print "the time of creating sim matrix is %.5fs" % (t2 - t1)
    t1 = time.time()
    db = DBSCAN(eps=0.5, min_samples=5, metric='precomputed').fit(sim_matrix)
    t2 = time.time()
    print "the time of clustering is %.5fs" % (t2 - t1)
    # 构图
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=10)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)
    plt.title('SNN')
    plt.show()

    # dbscan聚类算法
    t1 = time.time()
    db = DBSCAN(eps=12, min_samples=15).fit(X)
    t2 = time.time()
    print t2 - t1
    # 构图
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=10)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('dbScan')
    plt.show()