from matplotlib import pyplot
from matplotlib.colors import Colormap
import pandas as pd
from pandas.plotting import scatter_matrix

filename = 'diabetes.csv'
data = pd.read_csv(filename)
scatter_matrix(data)
pyplot.show()