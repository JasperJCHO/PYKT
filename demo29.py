from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 0], [0, 1], [1, 2], [1, 4], [1, 8],
              [4, 2], [4, 4], [4, 0], [4, 6], [4, 7]])
kmeans = KMeans(n_clusters=2).fit(X)
print(kmeans)
print("cluster labels=%s" % kmeans.labels_)
print("predict several=", kmeans.predict([[0, 0], [5, 5], [3, 3]]))
print("centers=", kmeans.cluster_centers_)
print("kmean inertia={}".format(kmeans.inertia_))