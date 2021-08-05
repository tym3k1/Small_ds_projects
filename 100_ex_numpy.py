#1. Import the numpy package under the name np (★☆☆)
import numpy as np

#2. Print the numpy version and the configuration (★☆☆)
print(np.__version__) 
print(np.show_config)

#3. Create a null vector of size 10 (★☆☆)
zeros_vector = np.zeros(10)

#4. How to find the memory size of any array (★☆☆)
zeros_vector.size
zeros_vector.itemsize

#5. How to get the documentation of the numpy add function from the command line? (★☆☆)
#np.info()

#6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
zeros_vector[4] = 1

#7. Create a vector with values ranging from 10 to 49 (★☆☆)
vector_values = np.arange(10,50)

#8. Reverse a vector (first element becomes last) (★☆☆)
reverse_vector_values = vector_values[::-1]

#9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
matrix_3x3 = np.arange(9).reshape(3,3)

#10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)
x = np.array([1,2,0,0,4,0])
print(np.nonzero(x))

#11. Create a 3x3 identity matrix (★☆☆)
identity_matrix = np.eye(3)

#12. Create a 3x3x3 array with random values (★☆☆)
matrix_3x3x3 = np.random.rand(3,3,3)

#13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
matrix_10x10 = np.random.rand(10,10)
Mmax, Mmin = matrix_10x10.max(), matrix_10x10.min()

#14. Create a random vector of size 30 and find the mean value (★☆☆)
vector_30 = np.random.rand(30)
v30 = vector_30.mean()

#15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
new_2d = np.ones((4, 4))
new_2d[1:-1, 1:-1]=0

#16. How to add a border (filled with 0's) around an existing array? (★☆☆)
new_2d_2 = np.pad(new_2d, pad_width=1)

#17. What is the result of the following expression? (★☆☆)
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
""" nan
False
False
nan
True
False """

#18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
next_5x5 = np.zeros((5,5))
next_5x5 = np.diag([1,2,3,4], -1)

#19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
next_8x8 = np.zeros((8,8))
next_8x8[1::2,::2] = 1
next_8x8[::2,1::2] = 1

#20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)
print(np.unravel_index(99,(6,7,8)))

#21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
next_8x8_2 = np.array([[0, 1],[1,0]])
np.tile(next_8x8_2, (4,4))

#22. Normalize a 5x5 random matrix (★☆☆)
matrix_5x5_2 = np.random.rand(5,5)
(matrix_5x5_2-np.mean(matrix_5x5_2))/np.std(matrix_5x5_2)

#23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
color = np.dtype([("r", np.ubyte),
                  ("g", np.ubyte),
                  ("b", np.ubyte),
                  ("a", np.ubyte)])

#24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
matrix_5x3 = np.random.rand(5,5)
matrix_3x5 = np.random.rand(5,5)
np.dot(matrix_5x3, matrix_3x5)

#25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
oneD = np.arange(10)
oneD[(oneD>3) & (oneD<8)] *= -1

#26. What is the output of the following script? (★☆☆)
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
hint: np.sum

#27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
#No hints provided...

#28. What are the result of the following expressions? (★☆☆)
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
#No hints provided...

#29. How to round away from zero a float array ? (★☆☆)
#hint: np.uniform, np.copysign, np.ceil, np.abs, np.where
Z = np.random.uniform(-10,+10,10)
np.copysign(np.ceil(np.abs(Z)), Z)
#or
np.where(Z>0, np.ceil(Z), np.floor(Z))

#30. How to find common values between two arrays? (★☆☆)
d_arr1 = np.random.randint(0,10,10)
d_arr2 = np.random.randint(0,10,10)
np.intersect1d(d_arr1, d_arr2)

#31. How to ignore all numpy warnings (not recommended)? (★☆☆)
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0
# Back to sanity
_ = np.seterr(**defaults)
# Equivalently with a context manager
with np.errstate(all="ignore"):
    np.arange(3) / 0

#32. Is the following expressions true? (★☆☆)
np.sqrt(-1) == np.emath.sqrt(-1)
#imaginary number

#33. How to get the dates of yesterday, today and tomorrow? (★☆☆)
yesterday = np.datetime64('today') - np.timedelta64(1)
today     = np.datetime64('today')
tomorrow  = np.datetime64('today') + np.timedelta64(1)

#34. How to get all the dates corresponding to the month of July 2016? (★★☆)
#hint: np.arange(dtype=datetime64['D'])
np.arange('2016-06', '2016-07', dtype='datetime64[D]')

#35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
#hint: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=)
A = np.ones(3)*1
B = np.ones(3)*2
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)

#36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)
xd = np.random.uniform(0,10,10)

print(xd - xd%1)
print(xd // 1)
print(np.floor(xd))
print(xd.astype(int))
print(np.trunc(xd))

#37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
x = np.zeros((5,5))
x += np.arange(5)

#38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)
def generuj():
    for x in range(10):
        yield x
j = np.fromiter(generuj(), dtype=int)

#39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
ranging_0_1 = np.linspace(0,1,11,endpoint=False)[1:]
#startreadingcomprehension 

#40. Create a random vector of size 10 and sort it (★★☆)
random_vector = np.random.uniform(0,10,10)
np.sort(random_vector)

#41. How to sum a small array faster than np.sum? (★★☆)
np.add.reduce([10], initial=5)

#42. Consider two random array A and B, check if they are equal (★★☆)
a_jeden = np.arange(10)
b_jeden = np.arange(10)
np.allclose(a_jeden, b_jeden)
np.array_equal(a_jeden, b_jeden)


#43. Make an array immutable (read-only) (★★☆)
immutable = np.arange(10)
immutable.flags.writeable = False
immutable[0] = 1

#44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
#hint: np.sqrt, np.arctan2
matrix_10x2 = np.random.rand(10,2)
X,Y = matrix_10x2[:,0], matrix_10x2 [:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)

#45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
vector_10 = np.random.rand(10)
argmax.vector_10 = 0


#46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area (★★☆)
_z = np.zeros((5,5), [('x',float),('y',float)])
_z['x'], _z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))

#47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)
_A = np.array([1, 2, 3])
_B = np.array([4, 5, 6])
_C = 1/np.subtract.outer(_A,_B)
np.linalg.det(_C)


#48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)
""" for dtype in [np.int8, np.int32, np.int64]:
       print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps) """

#49. How to print all the values of an array? (★★☆)
np.set_printoptions(threshold=float("inf"))
set_print = np.zeros((25,25))
print(set_print)

#50. How to find the closest value (to a given scalar) in a vector? (★★☆)
value = np.arange(100)
closest = np.random.uniform(0,100)
idx = (np.abs(value-closest)).argmin()
value[idx]

#51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)
""" Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z) """

#52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)
""" Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D) """

53. How to convert a float (32 bits) array into an integer (32 bits) in place?
 Z = (np.random.rand(10)*100).astype(np.float32)
Y = Z.view(np.int32)
print(Y)
Y[:] = Z


#54. How to read the following file? (★★☆)
#When dtype = null, then values equal nan
from io import StringIO
# Fake file
s = StringIO('''1, 2, 3, 4, 5

                6,  ,  , 7, 8

                 ,  , 9,10,11
''')
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)


#55. What is the equivalent of enumerate for numpy arrays? (★★☆)
enu = np.arange(9).reshape(3,3)
for x, index in np.ndenumerate(enu):
    print(index, x)
for index in np.ndindex(enu.shape):
    print( index, enu[index])

#56. Generate a generic 2D Gaussian-like array (★★☆)
J, K = np.meshgrid(np.linspace(-1,1,5), np.linspace(-1,1,5))
De = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (De-mu)**2 / ( 2.0 * sigma**2 ) ) )

#57. How to randomly place p elements in a 2D array? (★★☆)
n = 10
p = 3
zet = np.zeros((n,n))
np.put(zet, np.random.choice(range(n*n), p, replace=False),1)

#58. Subtract the mean of each row of a matrix (★★☆)
substr = np.random.rand(5, 10)
prt = substr-np.mean(substr, axis=1, keepdims=True)

#59. How to sort an array by the nth column? (★★☆)
Z = np.random.randint(0,10,(3,3))
Z[Z[:,1].argsort()]


#60. How to tell if a given 2D array has null columns? (★★☆)
#hint: any, ~
Z = np.random.randint(0,3,(3,10))
#print((~Z.any(axis=0)).any())

#61. Find the nearest value from a given value in an array (★★☆)
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]

62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
hint: np.nditer

63. Create an array class that has a name attribute (★★☆)
hint: class method

64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)
hint: np.bincount | np.add.at

65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)
hint: np.bincount

66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)
hint: np.unique

67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)
hint: sum(axis=(-2,-1))

68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (★★★)
hint: np.bincount

69. How to get the diagonal of a dot product? (★★★)
hint: np.diag

70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)
hint: array[::4]

71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)
hint: array[:, :, None]

72. How to swap two rows of an array? (★★★)
hint: array[[]] = array[[]]

73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles (★★★)
hint: repeat, np.roll, np.sort, view, np.unique

74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)
hint: np.repeat

75. How to compute averages using a sliding window over an array? (★★★)
hint: np.cumsum

76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)
hint: from numpy.lib import stride_tricks

77. How to negate a boolean, or to change the sign of a float inplace? (★★★)
hint: np.logical_not, np.negative

78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)
No hints provided...

79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)
No hints provided...

80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary) (★★★)
hint: minimum maximum

81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
hint: stride_tricks.as_strided

82. Compute a matrix rank (★★★)
hint: np.linalg.svd

83. How to find the most frequent value in an array?
hint: np.bincount, argmax

84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)
hint: stride_tricks.as_strided

85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)
hint: class method

86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)
hint: np.tensordot

87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)
hint: np.add.reduceat

88. How to implement the Game of Life using numpy arrays? (★★★)
No hints provided...

89. How to get the n largest values of an array (★★★)
hint: np.argsort | np.argpartition

90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)
hint: np.indices

91. How to create a record array from a regular array? (★★★)
hint: np.core.records.fromarrays

92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)
hint: np.power, *, np.einsum

93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)
hint: np.where

94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
No hints provided...

95. Convert a vector of ints into a matrix binary representation (★★★)
hint: np.unpackbits

96. Given a two dimensional array, how to extract unique rows? (★★★)
hint: np.ascontiguousarray | np.unique

97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)
hint: np.einsum

98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?
hint: np.cumsum, np.interp

99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)
hint: np.logical_and.reduce, np.mod

100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
hint: np.percentile





