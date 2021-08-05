#Univariate Histograms
from matplotlib import pyplot
import pandas as pd

#From shape of bins
#get feeling whether attribute is gaussian
#skewed or has exponential distribution
#also help se possible outliers
filename = 'diabetes.csv'
data = pd.read_csv(filename)
data.hist()
pyplot.show()