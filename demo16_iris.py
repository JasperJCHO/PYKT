import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
print(dir(iris))
labels = iris.feature_names
print(labels)
X = iris.data
species = iris.target
counter = 1
for i in range(0, 4):
    for j in range(i + 1, 4):
        xData = X[:, i]
        yData = X[:, j]
        plt.scatter(xData, yData, c=species, cmap=plt.cm.Paired)
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        plt.show()