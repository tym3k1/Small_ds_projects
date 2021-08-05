#feature extraction with univare statistical test
#used to select those features that have 
#strongest relationship with output
#chi-squared for non-negative features
#to select 4 of the best

import pandas as pd
import numpy as np
from pandas.core.construction import array
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#load data
fn = 'diabetes.csv'
dataframe = pd.read_csv(fn)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
#feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
#summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
#summarize selected features
print(features[0:5,:])
#2,5,6,8