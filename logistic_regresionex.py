import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn import model_selection

df = pd.read_csv("W:/vscode/Machine-Learning/HR_comma_sep.csv")

left = df[df.left == 1]
#print(left.shape)

retained = df[df.left == 0]
#print(retained.shape)

qq = df.groupby("left").mean(numeric_only=True)

#chart = pd.crosstab(df.salary,df.left)
#chart.plot(kind="bar")

#chart2 = pd.crosstab(df.Department,df.left)
#chart2.plot(kind="bar",figsize=(10,8))
plt.show()


subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

salary_dummys = pd.get_dummies(subdf.salary)
df_with_dummys = pd.concat([subdf,salary_dummys],axis="columns")
df_with_dummys.drop("salary",axis="columns", inplace=True)
#print(df_with_dummys)

x = df_with_dummys
y = df.left
#in case if we wanna predict to a specific value like model.predict([[0.11,310,0,0,1,0]]) bad satif level, a lot of working hours, 0,1,0 -> low salary so more likely to leave

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,train_size=0.3)
model = linear_model.LogisticRegression()

# model.fit(x_train,y_train)

# print(model.predict(x_test))
# print(model.score(x_test,y_test))

# model.fit(x,y)
# print(model.predict([[0.11,310,0,0,1,0]]))

