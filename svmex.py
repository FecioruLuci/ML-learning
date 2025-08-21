import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn import model_selection

digits = load_digits()
#print(digits.target_names)
# 'DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names'
df = pd.DataFrame(digits.data, digits.target)
df["target"] = digits.target

y = digits.target
x_train, x_test, y_train, y_test = model_selection.train_test_split(digits.data,y,train_size=0.2)

model = SVC(kernel="rbf",gamma="scale",C=4)
pred = model.fit(x_train,y_train)

print(model.score(x_test,y_test))



