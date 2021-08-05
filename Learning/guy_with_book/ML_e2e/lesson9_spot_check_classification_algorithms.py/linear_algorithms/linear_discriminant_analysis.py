#Linear Discriminant Analysis or LDA 
#is a statistical technique for binary and multiclass
#classification. It too assumes a Gaussian distribution


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
filename = 'diabetes.csv'
dataframe = pd.read_csv(filename)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=None)
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())