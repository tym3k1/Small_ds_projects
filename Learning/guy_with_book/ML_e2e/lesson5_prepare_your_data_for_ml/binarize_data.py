import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer
#threholding - probabilities u want make crisp val & feature eng. to sumting meaningful
# thresholding mean "próg" - "progowanie obrazu"
filename = 'diabetes.csv'
dataframe = pd.read_csv(filename)
array = dataframe.values
#separate array into input and output componets
X = array[:,0:8]
Y = array[:,8]
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
#summarize
np.set_printoptions(precision=3)
print(binaryX[0:5,:])