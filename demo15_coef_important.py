from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=511)
print(X.shape)
print(y.shape)
model = LinearRegression()
model.fit(X, y)
importance = model.coef_
for index, value in enumerate(importance):
    print(f'Feature #{index} score:{value:.4f}')
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
#print(model.feature_names)