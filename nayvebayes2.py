import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("W:/vscode/Machine-Learning/spam.csv")
#print(df.groupby("Category").describe())

df["spam"] = df["Category"].apply(lambda x: 1 if x == "spam"  else 0)
#print(df.head())

x_train, x_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2)
v = CountVectorizer()
x_train_count = v.fit_transform(x_train)
#print(x_train_count.toarray()[:3])
#convert text into number

# model = MultinomialNB()
# model.fit(x_train_count,y_train)


emails = [
    "Hello bro wanna go out tonight?",
    "Up to 20% discount on parking. Dont miss the reward!"
]

# emails_count = v.transform(emails)
# print(model.predict(emails_count))

# x_test_count = v.transform(x_test)
# print(model.score(x_test_count,y_test))

from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("nb", MultinomialNB())
])
emails = [
    "Hello bro wanna go out tonight?",
    "Up to 20% discount on parking. Dont miss the reward!"
]
clf.fit(x_train,y_train)
print(clf.predict(emails))
print(clf.score(x_test,y_test))


