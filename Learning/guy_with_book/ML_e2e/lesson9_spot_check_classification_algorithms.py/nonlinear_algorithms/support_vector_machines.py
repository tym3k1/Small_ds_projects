#Support Vector Machines (or SVM) 
#seek a line that best separates two classes
#Those data instances that are closest to the line 
#that best separates the classes are called support vectors

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
filename = 'diabetes.csv'
dataframe = pd.read_csv(filename)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=None)
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())