from pandas import read_csv

PATH = "data/iris.data"
df1 = read_csv(PATH, header=None)
print(type(df1), df1.shape)
print(df1.head())
dataset = df1.values
print(type(dataset))
features = dataset[:, :4].astype(float)
labels = dataset[:, 4]
print(features)
print(labels)
print("-----------------------------------------")
print(set(labels))