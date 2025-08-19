import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import model_selection
from sklearn import metrics
import seaborn as sb

digits = load_digits()

digits.data[0]

plt.gray()
# for i in range(5):
#     plt.matshow(digits.images[i])
x_train, x_test, y_train, y_test = model_selection.train_test_split(digits.data, digits.target, test_size=0.2)
# print(len(x_train))
# print(len(x_test))
model = linear_model.LogisticRegression()

model.fit(x_train,y_train)
plt.matshow(digits.images[67])
x = model.predict([digits.data[67]])
# print(model.predict(digits.data[0:5]))
# plt.show()

y_predicted = model.predict(x_test)

cm = metrics.confusion_matrix(y_test,y_predicted)


plt.figure(figsize=(10,7))
sb.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
