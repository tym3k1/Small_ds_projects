import numpy as np
import pandas as pd
from sklearn import datasets
#########estimator in the class, takes arguments as the model parameters
from sklearn import svm

iris = datasets.load_iris()
digits = datasets.load_digits()

np.set_printoptions(threshold=float("inf"))
#print(digits.data)
#print(digits.target)
#print(digits.images[0])

clf = svm.SVC(gamma=0.001, C=100.)

##########selecting training data, without last digit
clf.fit(digits.data[:-1], digits.target[:-1])

#########Predicting using the last image
#########image from training set
#########what have best maches to last image 
clf.predict(digits.data[-1:])