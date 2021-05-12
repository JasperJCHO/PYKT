import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris.target)
print(iris.target_names)
df['species'] = np.array([iris.target_names[i] for i in iris.target])
seaborn.pairplot(df, hue='species')
plt.show()