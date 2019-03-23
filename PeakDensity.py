# coding=utf-8
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │       ───       │
#      │  ─┬┘       └┬─  │
#      │                 │
#      │       ─┴─       │
#      │                 │
#      └───┐         ┌───┘
#          │         │             神兽保佑
#          │         │            代码无BUG!
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘              
# author = 'Eric Chen'
# create_date = '2019/3/23'



import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from math import sqrt
import sys


def euclidian_distance(vec_a, vec_b):
    """
    计算向量a与向量b的欧氏距离
    :param vec_a:
    :param vec_b:
    :return: 欧氏距离
    """
    diff = vec_a - vec_b
    return sqrt(np.dot(diff, diff))


# 计算各点间距离、各点点密度大小
def get_point_density(X, min_distance):
    X_num = X.__len__()
    distance_matrix = np.zeros((X_num, X_num))
    density_list = np.zeros(X_num)

    # 计算得到各点间距离
    for i in range(X_num):
        for j in range(i):
            if distance_matrix[i][j] > 0:
                continue
            distance_matrix[i][j] = euclidian_distance(X[i], X[j])
            distance_matrix[j][i] = distance_matrix[i][j]

    # 计算得到各点的密度
    for i in range(X_num):
        density_list[i] = np.where(distance_matrix[i] < min_distance)[0].__len__()
    return distance_matrix, density_list


def get_min_distance_list(distance_matrix, density_list):
    """
    获得点密度大于自身的最近点的距离
    :param distance_matrix:
    :param density_list:
    :return:
    """
    density_list = np.array(density_list)
    min_distance_list = []
    for i in range(density_list.__len__()):
        greater_index = np.where(density_list > density_list[i])[0]
        if len(greater_index) == 0:
            min_distance_list.append(np.max(distance_matrix))
            continue
        min_distance = sys.maxsize

        for j in greater_index:
            if distance_matrix[i][j] > 0:
                min_distance = min(min_distance, distance_matrix[i][j])
        min_distance_list.append(min_distance)
    return min_distance_list


def find_outliers(min_distance_list, threshold=3):
    """
    :param min_distance_list:
    :return: outliers_index_list
    """
    min_distance_list = np.array(min_distance_list)
    mean = np.mean(min_distance_list)
    std = np.std(min_distance_list)
    z_score = np.abs((min_distance_list - mean) / std)

    return np.where(z_score > threshold)[0]


def cluster(outliers_index_list, distance_matrix):
    """
    :param outliers_index_list:
    :param distance_matrix:
    :return:
    """
    y_predict = []
    min_distance = sys.maxint
    predict = -1
    for i in range(len(distance_matrix)):
        for j in outliers_index_list:
            if distance_matrix[i][j] < min_distance:
                min_distance = distance_matrix[i][j]
                predict = j
        y_predict.append(predict)
        min_distance = sys.maxint
    return y_predict


def get_picture(X, y, density_list, min_distance_list, outliers_index_list, y_predict):
    # 创建Figure
    fig = plt.figure()
    # 用来正常显示中文标签
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 原始点的分布
    ax1 = fig.add_subplot(221)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
    plt.title("origin data distribute")
    for i in outliers_index_list:

        plt.annotate("outlier", xy=(X[i, 0], X[i, 1]), xytext=(X[i, 0] + 1.5, X[i, 1] + 1.5),
                     arrowprops=dict(arrowstyle="->", facecolor='r'))
    plt.sca(ax1)

    ax2 = fig.add_subplot(222)
    plt.scatter(density_list, min_distance_list, c=y, s=10)
    plt.title("outliers distribute")
    for i in outliers_index_list:
        plt.annotate("outlier", xy=(density_list[i], min_distance_list[i]),
                     xytext=(density_list[i] + 1.5, min_distance_list[i] + 1.5),
                     arrowprops=dict(arrowstyle="->", facecolor='r'))
    plt.sca(ax2)


    # 聚类后分布
    ax3 = fig.add_subplot(223)
    plt.scatter(X[:, 0], X[:, 1], c=y_predict, s=10)
    plt.title("predict data distribute")
    plt.sca(ax3)

    plt.show()


if __name__ == '__main__':
    r = 1  # 邻域半径
    points_number = 500  # 随机点个数
    center_num = 3
    # 随机生成点坐标
    X, y = ds.make_blobs(points_number, centers=center_num, cluster_std=1, n_features=2)


    # 计算各点间距离、各点点密度(局部密度)大小
    distance_matrix, density_list = get_point_density(X, r)
    # 得到各点的聚类中心距离
    min_distance_list = get_min_distance_list(distance_matrix, density_list)
    outliers_index_list = find_outliers(min_distance_list, 2)
    y_predict = cluster(outliers_index_list, distance_matrix)
    # 画图
    get_picture(X, y, density_list, min_distance_list, outliers_index_list, y_predict)
