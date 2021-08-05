#ridge regression
#where the loss function is modified to
#minimize the complexity of the model measured
#sum squared value of the coefficient
#values (also called the L2-norm)

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(
    filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=None)
model = Ridge()
scoring = 'neg_mean_squared_error'
results = cross_val_score(
    model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())