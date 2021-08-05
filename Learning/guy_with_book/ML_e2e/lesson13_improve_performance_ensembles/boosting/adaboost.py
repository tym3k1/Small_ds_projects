#It generally works by weighting instances in 
#the dataset by how easy or difficult they are to classify, 
#allowing the algorithm to pay or less attention to them
#in the construction of subsequent models.
#AdaBooost classification

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
filename = 'diabetes.csv'
dataframe = pd.read_csv(filename)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 30
kfold = KFold(n_splits=10, random_state=None)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())