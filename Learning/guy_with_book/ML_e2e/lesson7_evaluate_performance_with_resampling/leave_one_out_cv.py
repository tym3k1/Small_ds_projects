#dataset into k = 1/10
#one outs


import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
fn = 'diabetes.csv'
dataframe = pd.read_csv(fn)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_folds = 10
seed = 7
loocv = LeaveOneOut()
model = LogisticRegression(max_iter=1000)
result = cross_val_score(model, X, Y, cv=loocv)
print(("Accuracy: %.3f%% (%.3f%%)") % (result.mean()*100.0, result.std()*100.0))