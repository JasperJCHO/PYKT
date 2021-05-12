import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(type(diabetes), dir(diabetes))
print('shape:',diabetes.data.shape, diabetes.target.shape)
print('feature_names',diabetes.feature_names)

dataForTest = -60
data_train = diabetes.data[:dataForTest]
print("data trained:", data_train.shape)
target_train = diabetes.target[:dataForTest]
print("target trained:", target_train.shape)

data_test = diabetes.data[dataForTest:]
print("data test:", data_test.shape)
target_test = diabetes.target[dataForTest:]
print("target test:", target_test.shape)

regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)
print(regression1.coef_)
print(regression1.intercept_)
print(f"model score={regression1.score(data_test, target_test)}")