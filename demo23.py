from sklearn import datasets, svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)
print(iris.data.shape, data.shape)
print(iris.data[0:5, ])
print(data[0:5, ])
datamax = data.max(axis=0) + 1
datamin = data.min(axis=0) - 1
n = 2000
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
print(len(X), len(Y))
svc = svm.SVC()
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
print(Z.shape)
plt.contour(X, Y, Z.reshape(X.shape))
plt.show()