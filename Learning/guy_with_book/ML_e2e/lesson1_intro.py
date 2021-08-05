#why python 2.7 :c
#numpy                   1.19.5
#matplotlib              3.4.1
#pandas                  1.2.4
#scipy                   1.6.3
#sklearn                 0.0xD
#joblib                  1.0.1

#1st lesson - think kinda repertory

#NUMPY CRASH COURSE - i know it
#define an array

import numpy as np
mylist = [[1, 2, 3],[4, 5, 6]]
myarray = np.array(mylist)
print(myarray.shape)

#acces data
print(("First row: %s") % myarray[0])
print(("Last row: %s") % myarray[-1])
print(("Specyfic row & column: %s") % myarray[0, 2])
print(("Whole column: %s") % myarray[:, 2])

#arythmetic
myarray1 = np.array([2, 2, 2])
myarray2 = np.array([3, 3, 3])
print(("Addition: %s") % (myarray1 + myarray2))
print(("Multiplication: %s") % (myarray1 * myarray2))

#MATPLOTLIB CRASH COURSE - I`m pretty sure, that I`m good at it
#basic lineplot
import matplotlib.pyplot as plt
#imprt..nump..

myarray_plt = np.array([1, 2, 3])
plt.plot(myarray_plt)
plt.xlabel('x axis')
plt.ylabel('xy axis')
#plt.show() #str8 line

#scatter - my professor was more demanding
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
plt.scatter(x, y)
#plt.show()

print('\n')
print('\n')
#PANDAS CRASH COURSE
#series
import pandas as pd
myarray_pandas = np.array([1, 2, 3])
rownames = ['a', 'b', 'c']
myseries = pd.Series(myarray_pandas, rownames)
print(myseries)

#acces to data in a series like numpy, or dict
print(myseries['a'])
print(myseries[0])

#dataframes
myarray_pandas_2 = np.array([[1, 2, 3],[4, 5, 6]])
rownames_2 = ['a', 'b']
colnames_2 = ['one', 'two', 'three']
mydataframe = pd.DataFrame(
    myarray_pandas_2, 
    index=rownames_2, 
    columns=colnames_2)

print(mydataframe)
#data can be index using column names
print('Method 1:')
print(('one column: %s') % mydataframe['one'])
print('Method 2:')
print(('one column: %s') % mydataframe.one)
