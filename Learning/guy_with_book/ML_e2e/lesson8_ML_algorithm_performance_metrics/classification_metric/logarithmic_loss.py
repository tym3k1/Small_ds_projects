#evaluating the predictions of probabilities
#of membership to a given class

#most common evaluation metric for classification problems

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
fn = 'diabetes.csv'
dataframe = pd.read_csv(fn)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=None)
model = LogisticRegression()
scoring = 'neg_log_loss'
result = cross_val_score(
    model, X, Y, cv=kfold, scoring=scoring)
print(("Accuracy: %.3f%% (%.3f%%)") 
    % (result.mean(), result.std()))