import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import decomposition
from sklearn import cross_validation
try:
	file_object = open("LeukemiaDataSet3.dat")
	all_lines = file_object.readlines()
	all_data = []
	for line in all_lines:
		all_data.append(line.strip().split("  "))

	for line in all_data:
		for i in range(0, len(line)):
			line[i] = float(line[i])


	df = pd.DataFrame(all_data)

	n_components = (int)((len(all_data[0])*10)**(0.5))
	#n_components = 10
	pca = decomposition.PCA(n_components=n_components,)
	#print df.iloc[:,1:len(all_data[0])-1]

	pca.fit(df.iloc[:,1:len(all_data[0])-1])
	X = pca.transform(df.iloc[:,1:len(all_data[0])-1])
	for line in X:
		for i in range(0, len(line)):
			line[i] = float(line[i])

	df2 = pd.DataFrame(X)

	clf1 = DecisionTreeClassifier(max_depth=4)
	clf2 = KNeighborsClassifier(n_neighbors=7)
	clf3 = SVC(kernel='rbf', probability=True)
	eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),('svc', clf3)],voting='soft', weights=[2, 1, 2])
	eclf2 = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),('svc', clf3)],voting='hard')

	# scores1 = cross_validation.cross_val_score(clf1, df.iloc[:,1:len(all_data[0])-1], np.ravel(df.iloc[:,0:1].values), cv=10)
	# scores2 = cross_validation.cross_val_score(clf2, df.iloc[:,1:len(all_data[0])-1], np.ravel(df.iloc[:,0:1].values), cv=10)
	# scores3 = cross_validation.cross_val_score(clf3, df.iloc[:,1:len(all_data[0])-1], np.ravel(df.iloc[:,0:1].values), cv=10)
	# scores4 = cross_validation.cross_val_score(eclf, df.iloc[:,1:len(all_data[0])-1], np.ravel(df.iloc[:,0:1].values), cv=10)
	# scores5 = cross_validation.cross_val_score(eclf2, df.iloc[:,1:len(all_data[0])-1], np.ravel(df.iloc[:,0:1].values), cv=10)

	scores1 = cross_validation.cross_val_score(clf1, df2.iloc[:,0:len(X[0])-1], np.ravel(df.iloc[:,0:1].values), cv=8)
	scores2 = cross_validation.cross_val_score(clf2, df2.iloc[:,0:len(X[0])-1], np.ravel(df.iloc[:,0:1].values), cv=8)
	scores3 = cross_validation.cross_val_score(clf3, df2.iloc[:,0:len(X[0])-1], np.ravel(df.iloc[:,0:1].values), cv=8)
	scores4 = cross_validation.cross_val_score(eclf, df2.iloc[:,0:len(X[0])-1], np.ravel(df.iloc[:,0:1].values), cv=8)
	scores5 = cross_validation.cross_val_score(eclf2, df2.iloc[:,0:len(X[0])-1], np.ravel(df.iloc[:,0:1].values), cv=8)

	print scores1,scores2,scores3,scores4,scores5

	


finally:
	file_object.close()
