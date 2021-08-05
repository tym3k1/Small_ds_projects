from matplotlib import pyplot
import pandas as pd

#drawing a line of median
#box around 25% to 75%, the middle %50 of data
#spread dots outside data to
#show candidate outlier values(1.5 times grater than
# size of spread of mdl 50%)

filename = 'diabetes.csv'
data = pd.read_csv(filename)
data.plot(kind='box', subplots=True, 
        layout=(3,3), sharex=False, sharey=False)
pyplot.show()