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


from DrawPicture import *

from math import sqrt
import sys


class DensityPeak:

    def __init__(self, X, cutoff_distance):
        """
        构造函数
        :param X: 数据集
        :param cutoff_distance: 截断距离
        :return:
        """
        self.X = X
        self.cutoff_distance = cutoff_distance
        self.distance_matrix = self.get_distance_matrix()
        self.density_list = self.get_local_density()
        self.outliers_index_list = None
        self.min_distance_list = None

    def euclidian_distance(self, vec_a, vec_b):
        """
        计算向量a与向量b的欧氏距离
        :param vec_a:
        :param vec_b:
        :return: 欧氏距离
        """
        diff = vec_a - vec_b
        return sqrt(np.dot(diff, diff))

    def get_distance_matrix(self):
        """
        获得距离矩阵，matrix[i][j] 便是点i与点j的欧式距离
        :return: matrix
        """
        X = self.X
        X_num = X.__len__()
        distance_matrix = np.zeros((X_num, X_num))
        # 计算得到各点间距离
        for i in range(X_num):
            for j in range(i):
                if distance_matrix[i][j] > 0:
                    continue
                distance_matrix[i][j] = self.euclidian_distance(X[i], X[j])
                distance_matrix[j][i] = distance_matrix[i][j]
        return distance_matrix

    def get_local_density(self):
        """
        获得每个点的局部密度
        :return: density_list
        """
        X = self.X
        X_num = X.__len__()
        cutoff_distance = self.cutoff_distance
        density_list = np.zeros(X_num)
        for i in xrange(X_num):
            density_list[i] = np.where(self.distance_matrix[i] < cutoff_distance)[0].__len__()
        return density_list

    def get_min_distance_list(self):
        """
        get the shortest distance from any other data point that has a higher density value than xi
        :return: min_distance_list
        """
        distance_matrix = self.distance_matrix
        density_list = self.density_list
        density_list = np.array(density_list)
        min_distance_list = []
        for i in range(density_list.__len__()):
            greater_index = np.where(density_list >= density_list[i])[0]
            if len(greater_index) == 1:
                min_distance_list.append(np.max(distance_matrix))
                continue
            min_distance = sys.maxsize

            for j in greater_index:
                if distance_matrix[i][j] > 0:
                    min_distance = min(min_distance, distance_matrix[i][j])
            min_distance_list.append(min_distance)
        return min_distance_list

    def find_outliers(self, threshold=2):
        """
        通过求z-score，寻找一维数据集 min_distance_list 的离群点
        :param threshold: 判断是否维离群点的阈值
        :return: outliers_index_list 
        """
        min_distance_list = self.min_distance_list
        min_distance_list = np.array(min_distance_list)
        mean = np.mean(min_distance_list)
        std = np.std(min_distance_list)
        z_score = np.abs((min_distance_list - mean) / std)
        return np.where(z_score > threshold)[0]

    def point_assign(self):
        """
        每个点的类别标签和高于当前点密度的最近的点的标签一致。
        :return:
        """
        outliers_index_list = self.outliers_index_list
        distance_matrix = self.distance_matrix
        density_list = self.density_list
        sort_density_index = np.argsort(-density_list)  # 获得局部密度值的降序排列对应的index
        y_predict = np.zeros(density_list.__len__(), dtype=int)
        y_predict[outliers_index_list] = outliers_index_list
        for i in xrange(sort_density_index.__len__()):
            if y_predict[sort_density_index[i]] == 0:
                min_distance = sys.maxint
                for j in xrange(i):
                    d = distance_matrix[sort_density_index[i]][sort_density_index[j]]
                    if d < min_distance:
                        min_distance = d
                        y_predict[sort_density_index[i]] = y_predict[sort_density_index[j]]
        return y_predict

    def get_decision_graph(self, zcore_threshold=2):
        """
        获得decision graph, 并将图片保存。
        :param zcore_threshold: 判断是否为离群点的阈值
        :return:
        """
        self.min_distance_list = self.get_min_distance_list()
        self.outliers_index_list = self.find_outliers(zcore_threshold)
        get_picture(self.density_list, self.min_distance_list, save_path='./graph/decision_graph.png',
                    title="decision graph",
                    outlier_index=self.outliers_index_list)
        get_picture(self.X[:, 0], self.X[:, 1], save_path='./graph/origin_with_outliers.png', title="origin data",
                    outlier_index=self.outliers_index_list)

    def clustering(self, outliers_index_list=None):
        """
        获得聚类结果，并将图片保存
        :param outliers_index_list: 
        :return:
        """
        if outliers_index_list is not None:
            self.outliers_index_list = outliers_index_list
        y_predict = self.point_assign()

        get_picture(self.X[:, 0], self.X[:, 1], save_path='graph/clustering.png', title="clustering result",
                    color=y_predict, outlier_index=self.outliers_index_list)
