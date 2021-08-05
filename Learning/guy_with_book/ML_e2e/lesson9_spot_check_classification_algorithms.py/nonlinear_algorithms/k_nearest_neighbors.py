#The k-Nearest Neighbors algorithm (or KNN) 
#uses a distance metric to find the k most similar
#instances in the training data for a new instance 
#and takes the mean outcome of the neighbors
#as the prediction. 



import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
filename = 'diabetes.csv'
dataframe = pd.read_csv(filename)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=None)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())