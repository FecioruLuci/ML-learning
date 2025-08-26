import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, GaussianNB

wine = load_wine()

df = pd.DataFrame(wine.data,columns=wine.feature_names)
#print(df)

df["target"] = wine.target
#print(df.head())

x_train, x_test, y_train, y_test = train_test_split(wine.data, df.target,test_size=0.2)

model = MultinomialNB()
model.fit(x_train,y_train)

model.predict(x_test)

print(model.score(x_test,y_test))

model2 = GaussianNB()

model2.fit(x_train,y_train)
model2.predict(x_test)

print(f"Multi e cel de sus si gausian este: {model2.score(x_test,y_test)}")
