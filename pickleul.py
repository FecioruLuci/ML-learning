import pandas as pd
from sklearn import linear_model
import numpy as np
import pickle
import joblib



df = pd.read_csv("W:/vscode/Machine-Learning/preturi.csv")

model = linear_model.LinearRegression()

model.fit(df[["area"]],df.price)

coef = model.coef_
inter = model.intercept_

pred = model.predict([[5000]])
with open("model_pickle","wb") as f:
    pickle.dump(model,f)

with open("model_pickle", "rb") as f:
    modelul = pickle.load(f)

pm = modelul.predict([[5000]])
print(pm)


joblib.dump(model,"joblib_model")
jm = joblib.load("joblib_model")

print(f"{jm.predict([[5000]])}")


