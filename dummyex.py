import pandas as pd
import numpy as np
from sklearn import linear_model


df = pd.read_csv("W:/vscode/Machine-Learning/carprices.csv")

model = linear_model.LinearRegression()

dummyes = pd.get_dummies(df["Car Model"])
#dummyes = dummyes.astype(int)

conq = pd.concat([df,dummyes],axis="columns")

final = conq.drop(["Car Model","Mercedez Benz C class"],axis="columns")

X = final.drop(["Sell Price($)"],axis="columns")
Y = final["Sell Price($)"]

#model.fit(X,Y)
#print(X)
pred = model.predict([[45000,4,False,False]])
pred2 = model.predict([[86000,7,False,True]])

print(pred)
print(pred2)
acc = model.score(X,Y)
print(acc)



