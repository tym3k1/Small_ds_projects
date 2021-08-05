from matplotlib import pyplot
import pandas as pd

#cleaner than in histograms

filename = 'diabetes.csv'
data = pd.read_csv(filename)
data.plot(kind='density', subplots=True, 
        layout=(3,3), sharex=False)
pyplot.show()