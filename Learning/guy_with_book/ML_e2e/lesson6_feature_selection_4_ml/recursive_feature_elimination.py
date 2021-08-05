#attributes or comination of attrib
#contribute most to predicting the target attribute
#using RFE with log.reg.alg to select top 3 features

#featur extraction with RFE
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#load data
fn = 'diabetes.csv'
dataframe = pd.read_csv(fn)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
#feature extraction
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, 3)
fit = rfe.fit(X,Y)
print(fit.n_features_to_select)
print(fit.support_)
print(fit.ranking_)