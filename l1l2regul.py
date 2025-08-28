import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

df = pd.read_csv("W:/vscode/Machine-Learning/Melbourne_housing_FULL.csv")
#print(df.head(5))

good_cols = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']

df = df[good_cols]
#print(df.head(6))
fill_0 = ["Propertycount", "Distance", "Bedroom2", "Bathroom", "Car"]
df[fill_0] = df[fill_0].fillna(0)
df["BuildingArea"] = df['BuildingArea'].fillna(df['BuildingArea'].mean())
df["Landsize"] = df["Landsize"].fillna(df["Landsize"].mean())
df.dropna(inplace=True)
#print(df.isna().sum())

df = pd.get_dummies(df,drop_first=True)
x = df.drop(["Price"],axis="columns")
y = df["Price"]
#print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#normal regresion
# model = LinearRegression()
# model.fit(x_train,y_train)
# model.predict(x_test)
# print(model.score(x_test,y_test))
# print(model.score(x_train,y_train))
#l1 
# model = Lasso(alpha=50,max_iter=100, tol=0.1)
# model.fit(x_train,y_train)
# print(model.score(x_test,y_test))
# print(model.score(x_train,y_train))
#l2
model = Ridge()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.score(x_train,y_train))