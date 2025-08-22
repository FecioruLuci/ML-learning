import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = load_iris()
#print(dir(df))
#print(df)
#print(df.target_names)
x_train, x_test, y_train, y_test = train_test_split(df.data,df.target)

model = RandomForestClassifier(n_estimators=30)
model.fit(x_train,y_train)

pred = model.predict(x_test)

print(model.score(x_test,y_test))

