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
Create_Time = 2016/07/05
一趟聚类
"""

from ClusterUnit import *
from CaculateDistance import euclidian_distance


class OnePassCluster:
    def __init__(self, threshold, vector_list):
        self.threshold = threshold  # 一趟聚类的阀值
        self.vectors = np.array(vector_list)
        self.cluster_list = []  # 聚类后簇的列表
        self.clustering()

    def clustering(self):
        self.cluster_list.append(cluster_unit().add_node(0, self.vectors[0]))  # 初始新建一个簇
        for index in range(len(self.vectors))[1:]:
            for cluster in self.cluster_list:
                if euclidian_distance(vec_a=self.vectors[index], vec_b=cluster.centroid) < self.threshold:  # 距离小于阀值归于该簇
                    cluster.add_node(index, self.vectors[index])
                else:  # 新建一个簇
                    self.cluster_list.append((cluster_unit().add_node(index, self.vectors[index])))

    def print_result(self, with_label=None):
        """
        print出聚类结果
        :return:
        """
        print "the number of nodes %s" % len(self.vectors)
        print "the result of clustering:"
        print "the number of cluster %s" % len(self.cluster_list)
        for index, cluster in enumerate(self.cluster_list):
            print "----------------"
            print "cluster: %s " % index  # 簇的序号
            print cluster.node_list  # 该簇的节点列表
            if with_label is None:  #
                pass
            print "node num: %s" % cluster.node_num
