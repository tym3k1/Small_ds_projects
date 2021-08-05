#performance metric for binary classification
#An area of 1.0 represents a model
# that made all predictions perfectly.
# An area of 0.5 represents 
# a model that is as good as random

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
scoring = 'roc_auc'
result = cross_val_score(
    model, X, Y, cv=kfold, scoring=scoring)
print(("Accuracy: %.3f%% (%.3f%%)") 
    % (result.mean(), result.std()))

# You can see the AUC is relatively close
# to 1 and greater than 0.5,
# suggesting some skill in the predictions