#uses linear algebra to transform ds into compressed form
#know as data reduction technique
#u can chose the num of dimensions
#or principal components

#featur extraction with RFE
import pandas as pd
from sklearn.decomposition import PCA
#load data
fn = 'diabetes.csv'
dataframe = pd.read_csv(fn)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
#feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
#summarize comp
print(fit.explained_variance_ratio_)
print(fit.components_)