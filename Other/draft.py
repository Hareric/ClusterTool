# coding=utf-8
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
# centers = [[1, 1], [-1, -1], [1, -1]]
# centers_2 = [[3, 3], [-3, 3],[-3, -3]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.)
# X_2, l_2 = make_blobs(n_samples=750, cluster_std=0.5, centers=centers_2)
# X = np.concatenate((X, X_2))
X = np.load('dataSet/diff_density_data.npy')
plt.plot(X[:, 0], X[:, 1], 'o', markerfacecolor='y',
                 markeredgecolor='k', markersize=10)
plt.title("different density clusters")
plt.show()