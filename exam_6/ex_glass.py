import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

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
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),('svc', clf3)],voting='soft', weights=[2, 1, 2])
eclf2 = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),('svc', clf3)],voting='hard')

clf1.fit(train_text, train_lable)
clf2.fit(train_text, train_lable)
clf3.fit(train_text, train_lable)
eclf.fit(train_text, train_lable)
eclf2.fit(train_text, train_lable)
result1 = clf1.predict(test_text)
result2 = clf2.predict(test_text)
result3 = clf3.predict(test_text)
result4 = eclf.predict(test_text)
result5 = eclf2.predict(test_text)
error1 = 0
error2 = 0
error3 = 0
error4 = 0
error5 = 0
for i in range(len(result1)):
	if result1[i] != test_lable[i]:
		error1 = error1 + 1
	if result2[i] != test_lable[i]:
		error2 = error2 + 1
	if result3[i] != test_lable[i]:
		error3 = error3 + 1
	if result4[i] != test_lable[i]:
		error4 = error4 + 1
	if result5[i] != test_lable[i]:
		error5 = error5 + 1
error1 = (1- error1*0.1/len(result1))*100
error2 = (1- error2*0.1/len(result1))*100
error3 = (1- error3*0.1/len(result1))*100
error4 = (1- error4*0.1/len(result1))*100
error5 = (1- error5*0.1/len(result1))*100
print error1,error2,error3,error4,error5