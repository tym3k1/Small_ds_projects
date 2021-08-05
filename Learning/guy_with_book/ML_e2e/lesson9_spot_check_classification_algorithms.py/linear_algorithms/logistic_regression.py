#logistic reggresion
#assumes a Gaussian distribution for the numeric 
#input variables and can
#model binary classification problems.


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = 'diabetes.csv'
dataframe = pd.read_csv(filename)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=None)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())