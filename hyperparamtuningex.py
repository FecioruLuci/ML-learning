from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
import pandas as pd

digits = load_digits()

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'gaussianNB': {
        'model': GaussianNB(),
        'params': {}
    },
    'multinominalNB': {
        'model': MultinomialNB(),
        'params': {}
    },
    'treeclassifier': {
        'model': DecisionTreeClassifier(),
        'params': {}
    }
}
scores = []
for module_name, typee in model_params.items():
    clf = GridSearchCV(typee["model"],typee["params"],cv=5,return_train_score=False)
    clf.fit(digits.data,digits.target)
    scores.append({
        'model' : module_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
df = pd.DataFrame(scores)
print(df)
#as we can see svm is the most optimal for the digits dataset