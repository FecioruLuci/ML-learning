from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
import numpy as np

iris = load_iris()

x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data,iris.target,test_size=0.2)

model = linear_model.LogisticRegression(max_iter=200)
model.fit(x_train,y_train)

pred = model.predict(x_test)
print(pred)
print(metrics.accuracy_score(y_test,pred))


y_pred = model.predict(np.array([[5.1, 3.5, 1.4, 0.2]]))
print(iris.target_names[y_pred][0])

