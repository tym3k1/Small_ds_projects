import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(teturn_X_y=True)

clf = SVC()
clf.set_params(kernel='linear').fit(X,y)
##### >>>SVC(kernel='linear')

clf.predict(X[0:5])
##### >>>array([0, 0, 0, 0, 0])

clf.set_params(kernel='rbf').fit(X, y)
##### >>>SVC()

clf.predict(X[0:5])
##### >>>array([0, 0, 0, 0, 0])


###HYPERPARAMETERS OF EN ESTIMATOR
###CAN BE UPDATED
###fit() OWERWRITE WHAT HAS LERNED