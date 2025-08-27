import pandas as pd
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df["flower"] = iris.target
df["flower"] = df["flower"].apply(lambda x: iris.target_names[x])
#print(df[47:52])

x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.3)

model = svm.SVC(kernel="rbf",C=30,gamma="auto")
model.fit(x_train,y_train)
pred = model.predict(x_test)

#print(model.score(x_test,y_test))

#print(cross_val_score(svm.SVC(kernel="rbf",C=30,gamma="auto"),iris.data,iris.target,cv=5))

# val = [1,10,20]
# modele = ["rbf", "linear"]

# for x in val:
#     for model in modele:
#         print(f"For val={x}--{model}--{cross_val_score(svm.SVC(kernel=model,C=x,gamma="auto"),iris.data,iris.target)}")
#     print()

# clf = GridSearchCV(svm.SVC(gamma="auto"),{
#     "C": [1,10,20],
#     "kernel": ["rbf","linear"]
# },cv=5, return_train_score=False)

# clf.fit(iris.data,iris.target)
# #print(clf.cv_results_)
# df = pd.DataFrame(clf.cv_results_)
# print(df[["param_C","param_kernel","mean_test_score"]])

# clf = RandomizedSearchCV(svm.SVC(gamma="auto"),{
#     "C": [1,10,20],
#     "kernel": ["rbf","linear"]
# },cv=5, return_train_score=False,n_iter=2)

# clf.fit(iris.data,iris.target)
# df = pd.DataFrame(clf.cv_results_)
# print(df[["param_kernel","param_C","mean_test_score"]])




        