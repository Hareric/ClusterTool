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
Create_Time = 2016/07/04
簇
"""

import numpy as np


class ClusterUnit:
    def __init__(self):
        self.node_list = []  # 该簇存在的节点列表
        self.node_num = 0  # 该簇节点数
        self.centroid = None  # 该簇质心

    def add_node(self, node, node_vec):
        """
        为本簇添加指定节点，并更新簇心
        :param node_vec: 该节点的特征向量
        :param node: 节点
        :return: null
        """
        self.node_list.append(node)
        try:
            self.centroid = (self.node_num * self.centroid + node_vec) / (self.node_num + 1)  # 更新簇心
        except TypeError:
            self.centroid = np.array(node_vec) * 1.  # 初始化质心
        self.node_num += 1  # 节点数加1

    def remove_node(self, node):
        """
        移除本簇指定节点
        :param node: 节点
        :return: null
        """
        try:
            self.node_list.remove(node)
            self.node_num -= 1
        except ValueError:
            raise ValueError("%s not in this cluster" % node)  # 该簇本身就不存在该节点，移除失败
        '''更新质心 待完成'''

    def move_node(self, node, another_cluster):
        """
        将本簇中的其中一个节点移至另一个簇
        :param node: 节点
        :param another_cluster: 另一个簇
        :return: null
        """
        self.remove_node(node=node)
        another_cluster.add_node(node=node)
        '''更新质心 待完成'''


if __name__ == '__main__':
    cluster_unit = ClusterUnit()
    cluster_unit.add_node(1, [1, 1, 2])
    cluster_unit.add_node(5, [2, 1, 2])
    cluster_unit.add_node(3, [3, 1, 2])
    print cluster_unit.centroid
