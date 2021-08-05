#rescale data (between 0 and 1)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#scale bt 0-1, to set one type, gud4 regress, nn and alg us distance
filename = 'diabetes.csv'
dataframe = pd.read_csv(filename)
array = dataframe.values
#separate array into input and output componets
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledxX = scaler.fit_transform(X)
#wsummarize transformed data
np.set_printoptions(precision=3)
print(rescaledxX[0:5,:])