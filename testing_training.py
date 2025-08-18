import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import model_selection

df = pd.read_csv("W:/vscode/Machine-Learning/carprices2.csv")
df.head()
#pltt = plt.scatter(df[["Mileage"]],df[["Sell Price($)"]])
#pltt2 = plt.scatter(df[["Age(yrs)"]],df[["Sell Price($)"]])
X = df[["Mileage","Age(yrs)"]]
Y = df[["Sell Price($)"]]

x_train, x_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size=0.2)

clf = linear_model.LinearRegression()
clf.fit(x_train,y_train)
testu = clf.predict(x_test)
print(y_test)
print(testu)
print(clf.score(x_test,y_test))

