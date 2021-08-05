import numpy as np

def zad1():
    loc = np.array([[1,  6, 11],
    [2,  7, 12],
    [3,  8, 13],
    [4,  9, 14],
    [5, 10, 15]])
    loc2 = loc[2::2]
    print(loc2)

def zad2():
    a = np.arange(25).reshape(5, 5)
    b = np.array([1., 5, 10, 15, 20])
    c = a/b[:, np.newaxis]
    print(c)

def zad3():
    x1 = np.random.uniform(0,1,(3,10))
    dop = [a[np.abs(a-0.5).argmin()] for a in x1]
    print(dop)

