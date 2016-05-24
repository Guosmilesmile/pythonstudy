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
from sklearn.feature_selection import VarianceThreshold


def getTargetData(fielname):
    file_object  = open(fielname);
    all_text_lines = file_object.readlines()
    all_text = []
    train_text = []
    train_classfi = []
    for line in all_text_lines:
        all_text.append(line.strip().split(","))
    train_classfi  = all_text[0]
    for i in range(len(all_text)):
        if(i!=0):
            train_text.append(all_text[i])
    train_text = np.array(train_text)
    train_classfi = np.array(train_classfi)
    for i in range(len(train_text)):
        for j in range(len(train_text[0])):
            train_text[i][j] = float(train_text[i][j])
    train_classfi_number = []
    for i in range(len(train_classfi)):
        features = train_text[i]
        if train_classfi[i]=="lumina" :
            klass = 0
            train_classfi_number.append(klass)
        elif train_classfi[i]=="ERBB2" :
            klass = 1
            train_classfi_number.append(klass)
        elif train_classfi[i]=="basal" :
            klass = 2
            train_classfi_number.append(klass)
        elif train_classfi[i]=="normal" :
            klass = 3
            train_classfi_number.append(klass)
        elif train_classfi[i]=="cell_lines" :
            klass = 4
            train_classfi_number.append(klass)
    train_feature_name = []
    for i in range(len(train_text)):
        train_feature_name.append(i)
    return train_text.transpose(),train_classfi_number,train_classfi,train_feature_name

def AAC(listtest,listanswer):
    labels = list(set(listtest))
    total = 0
    for i in range(len(labels)):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(listanswer)):
            if(labels[i]==listanswer[j]):
                if(listanswer[j]==listtest[j]):
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if(labels[i]==listtest[j]):
                    fn = tn + 1
                else:
                    tn = fn + 1
        total = total + float(tp+tn)/(tp+tn+fp+fn)*100
    return total/len(labels)

# train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("Breast_train.data")
# test_text,test_classfi_number,test_classfi,test_feature_name = getTargetData("Breast_test.data")

# for i in range(len(train_text)):
#         for j in range(len(train_text[0])):
#             train_text[i][j] = float(train_text[i][j])
#             print type(train_text[i][j] )

# selector = VarianceThreshold()
# data = selector.fit_transform(train_text)
# index = selector.get_support(True)

# train = data
# test = []
# df = pd.DataFrame(test_text)
# for line in index:
# 	test.append(df[line])

X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
selector = VarianceThreshold()
selector.fit_transform(X)
print selector.get_support()
# clf = DecisionTreeClassifier(max_depth=4)
# clf = SVC(kernel='rbf', probability=True)
# clf.fit(data, train_classfi)
# result = clf.predict(test_text)

