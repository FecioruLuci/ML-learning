import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sb
digits = load_digits()

#print(dir(digits))

# for i in range(4):
#     plt.matshow(digits.images[i])
# plt.show()

df = pd.DataFrame(digits.data)
df["target"] = digits.target
#print(df.head())
# drop = df.drop(["target"], axis="columns")

x_train, x_test, y_train, y_test = model_selection.train_test_split(df.drop(["target"],axis="columns"),digits.target,test_size=0.2)
model = RandomForestClassifier(n_estimators=30)
model.fit(x_train,y_train)
pred = model.predict(x_test)
#print(model.score(x_test,y_test))

cm = confusion_matrix(y_test,pred)
#print(cm)

plt.figure(figsize=(10,7))
sb.heatmap(cm,annot=True)
plt.xlabel("predicted")
plt.ylabel("truth")
plt.show()



