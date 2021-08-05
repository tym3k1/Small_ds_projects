###########   Peek at your data
#view first 20 rows
import pandas as pd
from pandas._config.config import set_option
filename = "diabetes.csv"
data = pd.read_csv(filename)
peek = data.head(20)
print(peek)


###########   Demensions of your data
#its about data and performence
#you need good shape and size
shape = data.shape
print(shape)


###########   Data type for each attribute
#converting types to categorical or ordinal values
types = data.dtypes
print(types)


###########   Descriptive statistic
#count
#mean
#std
#min
#25%
#50%
#75%
#max
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print(description)


###########   Class distribution(CLASSIFICATION ONLY)
#know how balanced class values are

class_counts = data.groupby('Outcome').size()
print(class_counts)


###########   Correlations between attributes
#how they match each other
#kinda like 1 0 or -1, if 0 then no correlations
set_option('display.width', 100)
set_option('precision', 3)
#Preason`s Correlation Coefficeint
correlations = data.corr(method='pearson')
print(correlations)


###########   Skrew of univariate distributions
#using gussian to correct the skew
skew = data.skew()
print(skew)

########### TIPS TO REMEMBER
#REVIEW THE NUMBERS - JUST LOOK AT IT BY FEW MOMENTS
#ASK WHY - WHY THIS LOOK LIKE THIS, HOW THEY RELATE?
#WRITE DOWN IDEAS - JUST OPEN NOTEPAD OR TAKE PAPER
                    #AND WRITE YOUR IDEAS
