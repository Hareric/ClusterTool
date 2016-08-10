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
    # snn_sim_table = []  # 三元组表记录相似度矩阵, .[0]与.[1]的相似度为.[2]
    sim_matrix = np.zeros((samples_size, samples_size), dtype=int)  # snn相似度矩阵
    for i in range(samples_size):
        for j in range(samples_size)[i + 1:]:
            sim_value = np.intersect1d(knn_matrix[i], knn_matrix[j]).__len__()  # 计算样本1与样本2的共享近邻点数
            if sim_value >= t:  # 仅保留相似度不小于t的对象
                # snn_sim_table.append([i, j, sim_value])
                sim_matrix[i][j] = sim_value
                sim_matrix[j][i] = sim_value

    return sim_matrix


if __name__ == '__main__':
    # X = [[1, 2, 3], [2, 4, 5], [2, 3, 1], [4, 5, 1]]




    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler


    ##############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    X = StandardScaler().fit_transform(X)
    print X
    print X.shape
    t1 = time.time()
    sim_matrix = snn_sim_matrix(X, k=8)
    t2 = time.time()
    print t2 - t1
    ##############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=5, min_samples=3).fit(sim_matrix)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    ##############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
