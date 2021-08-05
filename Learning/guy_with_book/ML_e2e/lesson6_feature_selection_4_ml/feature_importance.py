#bagged decision trees like random forest & extra trees

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
#load data
fn = 'diabetes.csv'
dataframe = pd.read_csv(fn)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
#feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)