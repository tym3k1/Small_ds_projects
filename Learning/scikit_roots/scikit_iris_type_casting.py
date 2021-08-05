from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
clf = SVC()

#print(clf.fit(iris.data, iris.target)) >>>SVC()
clf.fit(iris.data, iris.target)

#print(list(clf.predict(iris.data[0:3]))) >>> [0, 0, 0]
list(clf.predict(iris.data[0:3]))

#print(clf.fit(iris.data, iris.target_names[iris.target])) >>>SVC()
clf.fit(iris.data, iris.target_names[iris.target])

#print(list(clf.predict(iris.data[:3]))) >>> ['setosa', 'setosa', 'setosa']
list(clf.predict(iris.data[:3]))


