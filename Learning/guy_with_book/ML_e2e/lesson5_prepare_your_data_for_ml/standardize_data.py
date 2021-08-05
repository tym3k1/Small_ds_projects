from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
#using gussian -mean 0 & deviation 1, line.reg - log.reg - lin.dis.ana
dataframe = pd.read_csv('diabetes.csv')
array = dataframe.values
#separate array into input and output componets
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
#summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])