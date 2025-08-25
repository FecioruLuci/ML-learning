import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("W:/vscode/Machine-Learning/titanic.csv")
# print(df.head())
df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis="columns",inplace=True)
target = df.Survived
inputs = df.drop(["Survived"],axis="columns")
dummies = pd.get_dummies(inputs.Sex)
inputs = pd.concat([inputs,dummies],axis="columns")
#print(inputs.head())
inputs.drop(["Sex"],axis="columns",inplace=True)
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
#print(inputs.head(10))

x_train, x_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

model = GaussianNB()
model.fit(x_train,y_train)
pred = model.predict(x_test)
# print(model.score(x_test,y_test))
# print(x_test[:10])
# print(y_test[:10])
#if y_test== 0 its male else its female

proba = model.predict_proba(x_test)
print(proba)