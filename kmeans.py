import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt


df = pd.read_csv("W:/vscode/Machine-Learning/incomee.csv")

#plt.scatter(df["Age"],df["Income($)"])
#plt.show()
km = KMeans(n_clusters=3)
y_pred = km.fit_predict(df[["Age","Income($)"]])
#print(y_pred)
df["clutter"] = y_pred

# df0 = df[df.clutter==0]
# df1 = df[df.clutter==1]
# df2 = df[df.clutter==2]
# plt.scatter(df0.Age,df0["Income($)"],color="green")
# plt.scatter(df1.Age,df1["Income($)"],color="red")
# plt.scatter(df2.Age,df2["Income($)"],color="black")
# plt.xlabel("Age")
# plt.ylabel("Income")

scaler = MinMaxScaler()
scaler.fit(df[["Income($)"]])
df["Income($)"] = scaler.transform(df[["Income($)"]])
scaler.fit(df[["Age"]])
df["Age"] = scaler.transform(df[["Age"]])

km2 = KMeans(n_clusters=3)
y_pred2 = km2.fit_predict(df[["Age","Income($)"]])
df["cluster"] = y_pred2
df.drop("clutter",axis="columns",inplace=True)
print(df)
df0 = df[df.cluster==0]
df1 = df[df.cluster==1]
df2 = df[df.cluster==2]
plt.scatter(df0.Age,df0["Income($)"],color="green")
plt.scatter(df1.Age,df1["Income($)"],color="red")
plt.scatter(df2.Age,df2["Income($)"],color="black")
plt.scatter(km2.cluster_centers_[:,0],km2.cluster_centers_[:,1],color="purple",marker="+")
plt.xlabel("Age")
plt.ylabel("Income")
#plt.show()
sse = []
k_rng = range(1,10)
for k in k_rng:
    km2 = KMeans(n_clusters=k)
    km2.fit(df[["Age","Income($)"]])
    sse.append(km2.inertia_)

#print(sse)

plt.xlabel("K")
plt.ylabel("Sum")
plt.plot(k_rng,sse)
plt.show()
