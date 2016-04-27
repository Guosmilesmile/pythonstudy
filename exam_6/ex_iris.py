import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import cluster
from sklearn.metrics.cluster import adjusted_rand_score



iris = datasets.load_iris()

k_means = cluster.KMeans(n_clusters=3,precompute_distances =True).fit(iris.data)
db = cluster.DBSCAN(algorithm ='ball_tree').fit(iris.data)
fa = cluster.AgglomerativeClustering(n_clusters=3).fit(iris.data);
ap = cluster.AffinityPropagation(convergence_iter = 50,preference=-20).fit(iris.data);
print adjusted_rand_score(iris.target,k_means.labels_)
print adjusted_rand_score(iris.target,db.labels_)
print adjusted_rand_score(iris.target,fa.labels_)
print adjusted_rand_score(iris.target,ap.labels_)