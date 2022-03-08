import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

data = pd.read_csv('wine.csv', delimiter=',')


#print(data.head())
#print(data.shape)
#print(data.describe())

#different scales
#standardize data later


###############split data
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

#print(X_train_scaled.mean(axis=0))
#print(X_train_scaled.std(axis=0))

X_test_scaled = scaler.transform(X_test)

#print(X_test_scaled.mean(axis=0))
#print(X_test_scaled.std(axis=0))


##############preproc steps
pipeline = make_pipeline(preprocessing.StandardScaler(),
                        RandomForestRegressor(n_estimators=100))

#print(pipeline.get_params())


#############declare hyperparameters to tune
hyperparemeters = {'randomforestregressor__max_features' : ['auto', 'sqrt'],
                    'randomforestregressor__max_depth': [None, 5, 3, 1]}

#############tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparemeters, cv=10)

#############fit and tune model
clf.fit(X_train, y_train)


##############efit on the entire training set
#print(clf.best_params_) #best parameteters
#print(clf.refit)

############## Evaluate model pipeline on test data
y_pred = clf.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

joblib.dump(clf, 'rf_regressor.pkl')