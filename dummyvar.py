import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing

df = pd.read_csv("W:/vscode/Machine-Learning/homeprices2.csv")
le = preprocessing.LabelEncoder()
model = linear_model.LinearRegression()
ohe = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")
# print(df)
dummies = pd.get_dummies(df.town)
conq = pd.concat([df,dummies],axis="columns")

final = conq.drop(["town","west windsor "],axis="columns")

x = final.drop(["price"],axis="columns")
y = final.price

model.fit(x,y)
#pred = model.predict([[2800,0,1]])
pred = model.predict([[3400,0,0]])
aqq = model.score(x,y)

dfle = df
dfle.town = le.fit_transform(dfle.town)

X = dfle[["town","area"]].values
Y = dfle.price

X = ohe.fit_transform(X)
X = X[:,1:]
model.fit(X,Y)
pred = model.predict([[1,0,2800]])
print(pred)