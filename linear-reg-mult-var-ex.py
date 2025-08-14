import pandas as pd
import numpy as np
from sklearn import linear_model
import math
from word2number import w2n

df = pd.read_csv("W:/vscode/Machine-Learning/hiring.csv")

reg = linear_model.LinearRegression()

df.experience = df.experience.fillna("zero")
df.experience = df.experience.apply(w2n.word_to_num)
median = math.floor(df.test_score.mean())
df.test_score = df.test_score.fillna(median)
reg.fit(df[["experience","test_score","interview_score",]],df.salary)
pred1 = reg.predict([[12,10,10]])
pred2 = reg.predict([[2,9,6]])
print(pred1)
print(pred2)


