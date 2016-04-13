from sklearn import decomposition
from sklearn import datasets
from sklearn import svm
from sklearn import cluster
import pylab as pl
iris = datasets.load_iris()
# pca = decomposition.PCA(n_components=3)
# pca.fit(iris.data)
# X = pca.transform(iris.data)
# #pl.scatter(X[:, 0], X[:, 1], c=iris.target)
# print X
# #pl.show()

# svc = svm.SVC(kernel='rbf')
# svc.fit(iris.data[:-1], iris.target[:-1])
# result = svc.predict(iris.data[-1:])
# print result
# print iris.target[-1:]

k_means = cluster.KMeans(3)
k_means.fit(iris.data)
print k_means.labels_[::10]
print iris.target[::10]