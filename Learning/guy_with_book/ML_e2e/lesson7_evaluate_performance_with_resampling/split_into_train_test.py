#dataset into 67/33%
#evaluate accuracy of log.reg. model


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
fn = 'diabetes.csv'
dataframe = pd.read_csv(fn)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print(("Accuracy: %.3f%%") % (result*100.0))
