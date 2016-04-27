import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import cluster
from sklearn.metrics.cluster import adjusted_rand_score
def getTargetData(fielname):
	try:
		file_object = open(fielname)
		train_data = []
		train_text = []
		train_lable = []
		all_lines = file_object.readlines()
		del all_lines[0]
		for line in all_lines:
			train_data.append(line.strip().split(", "))
		for i in xrange(len(train_data)):
			temp = []
			for j in xrange(len(train_data[i])):
				if j == len(train_data[i])-1:
					train_lable.append(train_data[i][j])
				else :
					temp.append((float)(train_data[i][j]))
			train_text.append(temp)
		return train_text,train_lable
	finally:
		file_object.close()

train_text,train_lable = getTargetData("glass.data")
test_text,test_lable = getTargetData("glass.test")


k_means = cluster.KMeans(n_clusters=7,precompute_distances =True).fit(train_text)
db = cluster.DBSCAN(min_samples =2,algorithm ='ball_tree').fit(train_text)
fa = cluster.AgglomerativeClustering(n_clusters=7).fit(train_text);
ap = cluster.AffinityPropagation(convergence_iter = 50,preference=-20).fit(train_text);
print (adjusted_rand_score(train_lable,k_means.labels_))
print adjusted_rand_score(train_lable,db.labels_)
print adjusted_rand_score(train_lable,fa.labels_)
print adjusted_rand_score(train_lable,ap.labels_)
