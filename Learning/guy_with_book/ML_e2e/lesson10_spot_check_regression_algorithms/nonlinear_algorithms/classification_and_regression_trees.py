#Decision trees or the Classification and Regression Trees (CART as they are known)
#use the training data to select the best points to 
#split the data in order to minimize a cost metric.


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(
    filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=None)
model = DecisionTreeRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(
    model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())