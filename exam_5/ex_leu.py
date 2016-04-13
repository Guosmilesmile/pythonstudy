import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import decomposition

try:
	file_object = open("LeukemiaDataSet3.dat")
	all_lines = file_object.readlines()
	all_data = []
	for line in all_lines:
		all_data.append(line.strip().split("  "))
	df = pd.DataFrame(all_data)
	n_components = (int)((len(all_data[0])*10)**(0.5))
	pca = decomposition.PCA(n_components=n_components,)
	print df.iloc[:,1:len(all_data[0])-1]
	pca.fit(df.iloc[:,1:len(all_data[0])-1])
	X = pca.transform(df.iloc[:,1:len(all_data[0])-1])
	print X.shape
finally:
	file_object.close()
