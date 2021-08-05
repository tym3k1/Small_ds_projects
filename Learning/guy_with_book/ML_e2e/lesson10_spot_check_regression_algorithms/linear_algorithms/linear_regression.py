#Linear regression assumes that input variables 
#have a Gaussian distribution
#input variables are relevant(istotne) to the output variable
#and that they are not highly correlated 
#with each other (a problem called collinearity)


# Linear Regression
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
dataframe = pd.read_csv(
    filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=None)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(
    model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
