import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
#rescaling row to length 1 - sparse dataset(lot of 0) - neu.net n& distance such k-nn
filename = 'diabetes.csv'
dataframe = pd.read_csv(filename)
array = dataframe.values
#separate array into input and output componets
X = array[:,0:8]
Y = array[:,8]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
#summarize transfored data
np.printoptions(precison=3)
print(normalizedX[0:5,:])