#Naive Bayes calculates the probability of each class
# and the conditional probability of each class
#given each input value

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
filename = 'diabetes.csv'
dataframe = pd.read_csv(filename)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=None)
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())