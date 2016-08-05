# # from pylab import *
# # from mpl_toolkits.mplot3d import Axes3D
# #
# # fig = figure()
# # ax = Axes3D(fig)
# # X = np.arange(-4, 4, 0.25)
# # Y = np.arange(-4, 4, 0.25)
# # X, Y = np.meshgrid(X, Y)
# # R = np.sqrt(X**2 + Y**2)
# # Z = np.sin(R)
# #
# # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
# #
# # show()
#
# import numpy
#
# from math import sqrt
#
# u = numpy.array([1, 2, 3])
# v = [2, 1, 3]
#
#
# def euclidean_distance2(u, v):
#     """
#     Returns the euclidean distance between vectors u and v. This is equivalent
#     to the length of the vector (u - v).
#     """
#     diff = u - v
#     return sqrt(numpy.dot(diff, diff))
#
#
# print euclidean_distance2(u, v)

import numpy as np
from sklearn.neighbors import KDTree
import pickle
np.random.seed(0)
X = np.random.random((30, 3))
# print X
# r = np.linspace(0, 1, 5)
tree = KDTree(X, leaf_size=2)
dist, ind = tree.query(X[0], k=10)
s = pickle.dumps(tree)
print s
# print ind
# print dist
