import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv("W:/vscode/Machine-Learning/homeprices.csv")
median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)


reg = linear_model.LinearRegression()
reg.fit(df[["area","bedrooms","age"]],df.price)
print(reg.coef_)
print(reg.intercept_)

#print(reg.predict([[3000,3,40]]))
print(137.25*3000+-26025*3+-6825*40+383724)
print(reg.predict([[2500,4,5]]))
