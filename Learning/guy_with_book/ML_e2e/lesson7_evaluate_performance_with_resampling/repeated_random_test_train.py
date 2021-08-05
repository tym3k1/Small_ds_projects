#dataset into 67/33%
#repeat 10time


import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
fn = 'diabetes.csv'
dataframe = pd.read_csv(fn)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
n_split = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(
    n_splits=n_split, test_size=test_size,random_state=seed)
model = LogisticRegression(max_iter=1000)
result = cross_val_score(model, X, Y, cv=kfold)
print(("Accuracy: %.3f%% (%.3f%%)") % (result.mean()*100.0, result.std()*100.0))