import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import cluster
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


#k_means = cluster.KMeans(n_clusters=7,precompute_distances =True).fit(train_text)


#db = cluster.DBSCAN(min_samples =20,algorithm ='ball_tree').fit(train_text)

#fa = cluster.FeatureAgglomeration().fit(train_text);
ap = cluster.AffinityPropagation(convergence_iter = 50).fit(train_text);

print ap.labels_
#print train_lable