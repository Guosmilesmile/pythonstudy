from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.datasets import load_boston
# -*- coding: utf-8 -*-
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

# Only needed for data generation and graphical output
from pylab import ion, ioff, figure, draw, contourf, clf, show, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from random import normalvariate
import pandas as pd
import numpy as np
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


# convert a supervised dataset to a classification dataset
def _convert_supervised_to_classification(supervised_dataset,classes):
    classification_dataset = ClassificationDataSet(supervised_dataset.indim,supervised_dataset.outdim,classes)
    
    for n in xrange(0, supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(n)[0], supervised_dataset.getSample(n)[1])

    return classification_dataset


# def generate_data1():
#     INPUT_FEATURES = 9216 
#     CLASSES = 5
#     file_object = open("iris.data.txt")
#     lines = file_object.readlines()
#     lines.pop()
#     for i in xrange(0, len(lines)):
#         lines[i] = lines[i].strip().split(",")
#     for line in lines:
#         for i in xrange(0, len(line)-1):
#             line[i] = float(line[i])
#     for line in lines:
#         if line[4] == 'Iris-setosa':
#             line[4] = 0
#         elif line[4] == 'Iris-versicolor':
#             line[4] = 1
#         elif line[4] == 'Iris-virginica':
#             line[4] = 2
#     alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
#     for line in lines:
#         features = line[0:3]
#         klass = line[4]
#         alldata.addSample(features, [klass])
#     return {'minX': 0, 'maxX': 1,
#             'minY': 0, 'maxY': 1, 'd': alldata}

def generate_data():
    INPUT_FEATURES = 100 
    CLASSES = 5
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("Breast_train.data")
    X = train_text
    Y = train_classfi_number
    names = train_feature_name
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    temp=sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
                 reverse=True)
    ss = []
    count = 0
    df = pd.DataFrame(train_text)
    for line in temp:
        count == 0
        count += 1
        ss.append(df[line[1]].values)
        if(count==100):
            break
    train_text = np.array(ss).transpose()
    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        if train_classfi[i]=="lumina" :
            klass = 0
            alldata.addSample(features, klass)
        elif train_classfi[i]=="ERBB2" :
            klass = 1
            alldata.addSample(features, klass)
        elif train_classfi[i]=="basal" :
            klass = 2
            alldata.addSample(features, klass)
        elif train_classfi[i]=="normal" :
            klass = 3
            alldata.addSample(features, klass)
        elif train_classfi[i]=="cell_lines" :
            klass = 4
            alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata,'index':temp}

def generate_Testdata(index):
    INPUT_FEATURES = 100 
    CLASSES = 5
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("Breast_test.data")
    temp=index
    ss = []
    count = 0
    df = pd.DataFrame(train_text)
    for line in temp:
        count == 0
        count += 1
        ss.append(df[line[1]].values)
        if(count==100):
            break
    train_text = np.array(ss).transpose()
    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        if train_classfi[i]=="lumina" :
            klass = 0
            alldata.addSample(features, klass)
        elif train_classfi[i]=="ERBB2" :
            klass = 1
            alldata.addSample(features, klass)
        elif train_classfi[i]=="basal" :
            klass = 2
            alldata.addSample(features, klass)
        elif train_classfi[i]=="normal" :
            klass = 3
            alldata.addSample(features, klass)
        elif train_classfi[i]=="cell_lines" :
            klass = 4
            alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata,'index':temp}


def perceptron(hidden_neurons=20, weightdecay=0.01, momentum=0.1):
    INPUT_FEATURES = 100
    CLASSES = 5
    HIDDEN_NEURONS = hidden_neurons
    WEIGHTDECAY = weightdecay
    MOMENTUM = momentum
    
    g = generate_data()
    alldata = g['d']
    testdata = generate_Testdata(g['index'])['d']
    #tstdata, trndata = alldata.splitWithProportion(0.25)
    #print type(tstdata)

    trndata = _convert_supervised_to_classification(alldata,CLASSES)
    tstdata = _convert_supervised_to_classification(testdata,CLASSES)
    trndata._convertToOneOfMany()  
    tstdata._convertToOneOfMany()
    fnn = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,verbose=True, weightdecay=WEIGHTDECAY,learningrate=0.01)
    result = 0;
    ssss = 0;
    for i in range(100):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),trndata['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
        out = fnn.activateOnDataset(tstdata)
        ssss = out
        out = out.argmax(axis=1)
        result = out
    df = pd.DataFrame(ssss)
    df.to_excel("breastout.xls")
    df = pd.DataFrame(result)
    df.insert(1,'1',tstdata['class'])
    df.to_excel("breast.xls")
    error = 0;
    for i in range(len(tstdata['class'])):
        if tstdata['class'][i] != result[i]:
            error = error+1
    print (len(tstdata['class'])-error)*1.0/len(tstdata['class'])*100
    print AAC(result,tstdata['class'])
    NetworkWriter.writeToFile(fnn, 'breast.xml')
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Add more options if you like
    parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                        default=100,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                        default=0.03,
                        help="weightdecay")
    parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                        default=0.1,
                        help="momentum")
    args = parser.parse_args()

    perceptron(args.hidden_neurons, args.weightdecay, args.momentum)
    # g = generate_data()
    # print g['d']
