# coding=utf-8
import sys
import os

root_path = os.getcwd()  # 获得当前py的根目录
sys.path.append(root_path + '/Other')  # 导入Other文件夹
from OnePassCluster import *
import k_means
from sklearn import preprocessing

if __name__ == '__main__':
    vectors = np.loadtxt('Other/dataSet/dim1024.txt')  # k=16 threshold=0.1927
    # vectors = np.loadtxt('Other/dataSet/g2-256-100.txt')  # 0.192717136469125
    # 归一化（Normalization）
    vectors = preprocessing.normalize(vectors)  # 归一化
    # print vectors
    k = 16
    t1 = time.time()
    cluster_result = np.array(k_means.kmeans(k=k, vectors=vectors))
    t2 = time.time()
    print "k-means spend time %.9fs" % ((t2 - t1) / 1000)
    print cluster_result
    for i in range(k):
        print i, np.where(cluster_result == i)[0]
        print '-------'

    o_p_c = OnePassCluster(t=0.1927, vector_list=vectors)
    o_p_c.print_result()
