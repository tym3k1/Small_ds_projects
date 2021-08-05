from matplotlib import pyplot
from matplotlib.colors import Colormap
import pandas as pd
import numpy as np

#Showing correlation
#linear and logistic regression
#can gave poor performence
#if highly correlated inpit var in data

filename = 'diabetes.csv'
data = pd.read_csv(filename)
data_top = data.head() 
correlations = data.corr()
#plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 9, 1)
""" 
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(data_top)
ax.set_yticklabels(data_top) 
"""
pyplot.show()