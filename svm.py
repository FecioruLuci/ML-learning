import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVC



iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["flower_name"] = df.target.apply(lambda x: iris.target_names[x])
#print(df.head())

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]
# plt.xlabel("sepal lenght(cm)")
# plt.ylabel("sepal width(cm)")
# plt.scatter(df0["sepal length (cm)"], df0["sepal width (cm)"],color= "green", marker="+")
# plt.scatter(df1["sepal length (cm)"], df1["sepal width (cm)"],color= "blue", marker=".")
plt.xlabel("petal lenght(cm)")
plt.ylabel("petal width(cm)")
plt.scatter(df0["petal length (cm)"], df0["petal width (cm)"],color= "green", marker="+")
plt.scatter(df1["petal length (cm)"], df1["petal width (cm)"],color= "blue", marker=".")
#plt.show()

x = df.drop(["target","flower_name"],axis="columns")

#print(x.head())

y = df.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,train_size=0.2)

model = SVC()
pred = model.fit(x_train,y_train)
print(model.score(x_test,y_test))
