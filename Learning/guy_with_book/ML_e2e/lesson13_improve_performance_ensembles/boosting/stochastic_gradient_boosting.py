# proving to be perhaps one of the best techniques 
# available for improving performance via ensembles.
# Stochastic Gradient Boosting Classification

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
filename = 'diabetes.csv'
dataframe = pd.read_csv(filename)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 100
kfold = KFold(n_splits=10, random_state=None)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())