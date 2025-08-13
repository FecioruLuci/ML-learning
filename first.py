import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("W:/vscode/income.csv")

reg = linear_model.LinearRegression()
reg.fit(df[["year"]],df.income)
plt.scatter(df.year,df.income,color="red", marker="+")
plt.xlabel("year",fontsize=20)
plt.ylabel("income",fontsize = 20)

# reg.predict([[3300]])
#plt.show()
# print(f"{reg.predict([[3300]])}")
# print(f"{reg.coef_}")
# print(f"{reg.intercept_}")
# print(135.78767123*3300 + 180616.43835616432)

# db = pd.read_csv("W:/vscode/areas.csv")
# db.head(3)
# reg = linear_model.LinearRegression()
# reg.fit(df[["area"]],df.price)
# price = reg.predict(db)
# db["prices"] = price
# print(db)
# db.to_csv("predicition.csv")
# plt.plot(df.area,reg.predict(df[["area"]]),color="blue")
# plt.show()




#exercise on a diferent csv file
pred = reg.predict([[2021]])
print(pred)