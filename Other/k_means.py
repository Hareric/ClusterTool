# coding:utf-8
import nltk
import matplotlib.pylab as pl
import numpy as np
import time


# K_MEANS
def kmeans(k, vectors):  # k_means聚类,k为簇数
    km = nltk.cluster.KMeansClusterer(num_means=k,
                                      distance=nltk.cluster.util.euclidean_distance)  # 利用欧式距离创建K-mean聚类er，K为5
    km.cluster(vectors)  # 利用各地区最高温度与最低温度聚类
    cluster_result = []  # 聚类结果
    for temperat in vectors:
        cluster_result.append(km.classify(temperat))  # 获得类别
    return cluster_result


if __name__ == '__main__':
    temperature_all_city = np.loadtxt('c2.txt', delimiter=",", usecols=(3, 4))  # 读取聚类特征
    xy = np.loadtxt('c2.txt', delimiter=",", usecols=(8, 9))  # 读取各地经纬度
    k = 5
    t1 = time.time()
    cluster_result = kmeans(k=k, vectors=temperature_all_city)
    t2 = time.time()
    print "spend time %.5fs" % ((t2 - t1) / 1000)
    fig, ax = pl.subplots()
    cmap = pl.get_cmap('jet', k)
    for i in range(len(xy)):
        ax.scatter(xy[i][0], xy[i][1], c=cluster_result[i], s=30, cmap=cmap, vmin=0, vmax=k)
    pl.show()
