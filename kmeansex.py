import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df = df.drop(["sepal length (cm)","sepal width (cm)"],axis="columns")

# print(df.head())

model = KMeans(n_clusters=3)
y_pred = model.fit_predict(df[["petal length (cm)","petal width (cm)"]])

df["cluster"] = y_pred
# print(df)

df0 = df[df.cluster==0]
df1 = df[df.cluster==1]
df2 = df[df.cluster==2]

plt.scatter(df0["petal length (cm)"],df0["petal width (cm)"],color="green")
plt.scatter(df1["petal length (cm)"],df1["petal width (cm)"],color="black")
plt.scatter(df2["petal length (cm)"],df2["petal width (cm)"],color="yellow")
#plt.show()

sse = []
K_rng = range(1,11)

for k in K_rng:
    model = KMeans(n_clusters=k)
    model.fit(df)
    sse.append(model.inertia_)

plt.xlabel("K")
plt.ylabel("SSE")
plt.plot(K_rng,sse)
plt.show()