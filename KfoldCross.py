import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# model = linear_model.LogisticRegression()
# model.fit(x_train,y_train)
#print(model.score(x_test,y_test))

# model = SVC()
# model.fit(x_train,y_train)
# #print(model.score(x_test,y_test))

# model = RandomForestClassifier()
# model.fit(x_train,y_train)
# #print(model.score(x_test,y_test))

model = KFold(n_splits=3)
# for train_index, test_index in model.split([1,2,3,4,5,6,7,8,9]):
    #print(train_index, test_index)

def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)

folds = StratifiedKFold(n_splits=3)
score_l = []
score_svc = []
score_fold = []

# for train_index, test_index in folds.split(digits.data,digits.target):
#     x_train, x_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index],digits.target[test_index]
#     score_l.append(get_score(LinearRegression(), x_train, x_test, y_train, y_test))
#     score_svc.append(get_score(SVC(), x_train, x_test, y_train, y_test))
#     score_fold.append(get_score(RandomForestClassifier(), x_train, x_test, y_train, y_test))


#print(get_score(linear_model.LogisticRegression(), x_train, y_train, x_test, y_test))
# print(score_l)
# print(score_svc)
# print(score_fold)
print(cross_val_score(LinearRegression(), digits.data, digits.target))
print(cross_val_score(SVC(), digits.data, digits.target))
print(cross_val_score(RandomForestClassifier(), digits.data, digits.target))