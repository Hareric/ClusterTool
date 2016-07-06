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
import time
import matplotlib.pylab as pl


class OnePassCluster:
    def __init__(self, threshold, vector_list):
        self.threshold = threshold  # 一趟聚类的阀值
        self.vectors = np.array(vector_list)
        self.cluster_list = []  # 聚类后簇的列表
        t1 = time.time()
        self.clustering()
        t2 = time.time()
        self.cluster_num = len(self.cluster_list)  # 聚类完成后 簇的个数
        self.spend_time = t2 - t1

    def clustering(self):
        self.cluster_list.append(ClusterUnit())  # 初始新建一个簇
        self.cluster_list[0].add_node(0, self.vectors[0])  # 将读入的第一个节点归于该簇
        for index in range(len(self.vectors))[1:]:
            min_distance = euclidian_distance(vec_a=self.vectors[0], vec_b=self.cluster_list[0].centroid)  # 与簇的质心的最小距离
            min_cluster_index = 0  # 最小距离的簇的索引
            for cluster_index, cluster in enumerate(self.cluster_list[1:]):  # 寻找距离最小的簇，记录下距离和对应的簇的索引
                distance = euclidian_distance(vec_a=self.vectors[index], vec_b=cluster.centroid)
                if distance < min_distance:
                    min_distance = distance
                    min_cluster_index = cluster_index + 1
            if min_distance < self.threshold:  # 最小距离小于阀值,则归于该簇
                self.cluster_list[min_cluster_index].add_node(index, self.vectors[index])
            else:  # 否则新建一个簇
                new_cluster = ClusterUnit()
                new_cluster.add_node(index, self.vectors[index])
                self.cluster_list.append(new_cluster)
                del new_cluster

    def print_result(self, label_dict=None):
        """
        print出聚类结果
        :param label_dict: 节点对应的标签字典
        :return:
        """
        print "************ clustering result ************"
        for index, cluster in enumerate(self.cluster_list):

            print "cluster: %s " % index  # 簇的序号
            print cluster.node_list  # 该簇的节点列表
            if label_dict is None:
                pass
            else:
                print " ".join([label_dict[n] for n in cluster.node_list])  # 若有提供标签字典，则输出该簇的标签
            print "node num: %s" % cluster.node_num
            print "----------------"
        print "the number of nodes %s" % len(self.vectors)
        print "the number of cluster %s" % self.cluster_num
        print "spend time %.5fs" % (self.spend_time / 1000)


if __name__ == '__main__':
    # 读取测试集
    temperature_all_city = np.loadtxt('Other/c2.txt', delimiter=",", usecols=(3, 4))  # 读取聚类特征
    xy = np.loadtxt('Other/c2.txt', delimiter=",", usecols=(8, 9))  # 读取各地经纬度
    f = open('Other/c2.txt', 'r')
    lines = f.readlines()
    zone_dict = dict(zip(range(len(xy)), [i.split(',')[1] for i in lines]))  # 读取地区并转化为字典
    f.close()

    # 构建一趟聚类器
    clustering = OnePassCluster(vector_list=temperature_all_city, threshold=9)
    clustering.print_result(label_dict=zone_dict)

    # 将聚类结果导出图
    fig, ax = pl.subplots()
    fig = zone_dict
    c_map = pl.get_cmap('jet', clustering.cluster_num)
    c = 0
    for cluster in clustering.cluster_list:
        for node in cluster.node_list:
            ax.scatter(xy[node][0], xy[node][1], c=c, s=30, cmap=c_map, vmin=0, vmax=clustering.cluster_num)
        c += 1
    pl.show()
