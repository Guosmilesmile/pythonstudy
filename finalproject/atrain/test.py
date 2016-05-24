# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:07:03 2016

@author: Moon Xu
"""


print(__doc__)


import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


#读取训练集
with open('GCM_train.data','r') as f:
    data=f.readlines()
    dataSet=[]
    labels=data[0].strip('\n').split(',')
    for line in data[1:]:
        line=line.strip('\n')
        y=line.split(',') 
        for i in range(0,np.size(y)):
            y[i]=float(y[i])
        dataSet.append(y)
        
dataSet_train=np.array(dataSet).T
labels_train=np.array(labels)


#读取测试集
with open('GCM_test.data','r') as f:
    data=f.readlines()
    dataSet=[]
    labels=data[0].strip('\n').split(',')
    for line in data[1:]:
        line=line.strip('\n')
        y=line.split(',') 
        for i in range(0,np.size(y)):
            y[i]=float(y[i])
        dataSet.append(y)
        
dataSet_test=np.array(dataSet).T
labels_test=np.array(labels)



# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
clf4 = LogisticRegression(random_state=123)
clf5 = RandomForestClassifier(random_state=123)
clf6 = GaussianNB()



clf1.fit(dataSet_train, labels_train)
clf2.fit(dataSet_train, labels_train)
clf3.fit(dataSet_train, labels_train)
clf4.fit(dataSet_train, labels_train)
clf5.fit(dataSet_train, labels_train)
clf6.fit(dataSet_train, labels_train)


#输出基分类器正确率 
answer = clf1.predict(dataSet_test)  
correctRatio1=np.mean( answer == labels_test)
print "DecisionTree average accuary: %.2f%%" % (correctRatio1 * 100)

answer = clf2.predict(dataSet_test)  
correctRatio2=np.mean( answer == labels_test)
print "KNeighbors average accuary: %.2f%%" % (correctRatio2 * 100)

answer = clf3.predict(dataSet_test)  
correctRatio3=np.mean( answer == labels_test)
print "SVC average accuary: %.2f%%" % (correctRatio3 * 100) 

answer = clf4.predict(dataSet_test)  
correctRatio4=np.mean( answer == labels_test)
print "LogisticRegression average accuary: %.2f%%" % (correctRatio4 * 100) 

answer = clf6.predict(dataSet_test)  
correctRatio6=np.mean( answer == labels_test)
print "GaussianNB average accuary: %.2f%%" % (correctRatio6 * 100) 


answer = clf5.predict(dataSet_test)  
correctRatio5=np.mean( answer == labels_test)

#集成五种基分类器

eclf = VotingClassifier(estimators=[ ('knn', clf2), 
                                    ('svc', clf3)],
                        voting='soft',
                        weights=[1,3])
                        
'''
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3),('lr', clf4), 
                                    ('gnb', clf6)],
                        voting='soft',
                        weights=[correctRatio1,correctRatio2,correctRatio3,
                                 correctRatio4,correctRatio6])
'''

eclf.fit(dataSet_train, labels_train)

#输出集成分类器准确率
answer = eclf.predict(dataSet_test)  
correctRatio=np.mean( answer == labels_test)
print "\nVoting average accuary: %.2f%%" % (correctRatio * 100) 
print "RandomForest average accuary: %.2f%%" % (correctRatio5 * 100) 
