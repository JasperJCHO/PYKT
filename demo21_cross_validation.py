import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
data = iris.data
target = iris.target

logisticRegression1 = LogisticRegression()

classifiers = [logisticRegression1]

for classifier in classifiers:
    score = model_selection.cross_val_score(classifier, data, target, cv=3)
    print(f"classifier:{classifier} score={score}")