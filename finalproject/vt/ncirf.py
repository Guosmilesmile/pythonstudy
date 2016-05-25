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
from pybrain.tools.customxml.networkreader import NetworkReader
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
        all_text.append(line.strip().split(" "))
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
        if train_classfi[i]=="1" :
            klass = 0
            train_classfi_number.append(klass)
        elif train_classfi[i]=="2" :
            klass = 1
            train_classfi_number.append(klass)
        elif train_classfi[i]=="3" :
            klass = 2
            train_classfi_number.append(klass)
        elif train_classfi[i]=="4" :
            klass = 3
            train_classfi_number.append(klass)
        elif train_classfi[i]=="5" :
            klass = 4
            train_classfi_number.append(klass)
        elif train_classfi[i]=="6" :
            klass = 5
            train_classfi_number.append(klass)
        elif train_classfi[i]=="7" :
            klass = 6
            train_classfi_number.append(klass)
        elif train_classfi[i]=="8" :
            klass = 7
            train_classfi_number.append(klass)
        elif train_classfi[i]=="9" :
            klass = 8
            train_classfi_number.append(klass)
    train_feature_name = []
    for i in range(len(train_text)):
        train_feature_name.append(i)
    return train_text.transpose(),train_classfi_number,train_classfi,train_feature_name



def getIndexData(data,index):
    df = pd.DataFrame(data)
    result = []
    for line in index:
        result.append(df[line].values)
    return np.array(result).transpose()
    
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

def AUC(listtest,listanswer):
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
                    fn = fn + 1
                else:
                    tn = tn + 1
        total = total + float(tp)/(tp+fn)*100+float(tn)/(fp+tn)*100
    return total/len(labels)/2


def Fscore(listtest,listanswer):
    labels = list(set(listtest))
    pre1=0;pre2=0;rec=0
    for i in range(len(labels)):
        tp = 0;tn = 0;fp = 0;fn = 0
        for j in range(len(listanswer)):
            if(labels[i]==listanswer[j]):
                if(listanswer[j]==listtest[j]):
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if(labels[i]==listtest[j]):
                    fn = fn + 1
                else:
                    tn = tn + 1
        pre1 = pre1 + tp
        pre2 = pre2 + tp + fp
        rec = rec + tp + fn
    pre=float(pre1)/pre2
    rec=float(pre1)/rec
    return 2*pre*rec/(pre+rec)*100
# convert a supervised dataset to a classification dataset
def _convert_supervised_to_classification(supervised_dataset,classes):
    classification_dataset = ClassificationDataSet(supervised_dataset.indim,supervised_dataset.outdim,classes)
    
    for n in xrange(0, supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(n)[0], supervised_dataset.getSample(n)[1])

    return classification_dataset


def generate_data():
    index = [1623, 1009, 1689, 791, 201, 946, 1393, 2080, 858, 766, 637, 3408, 2621, 2851, 2094, 1225, 5417, 1258, 1180, 5347, 2573, 1723, 4361, 23, 3245, 2287, 1079, 808, 2590, 2875, 4129, 3279, 2459, 2585, 601, 4363, 5797, 4708, 682, 50, 955, 2468, 4286, 904, 980, 2786, 2523, 4437, 2021, 2072, 490, 183, 4541, 2300, 889, 6070, 2884, 806, 4103, 3404, 2371, 503, 4063, 3953, 3144, 1954, 1837, 4769, 4345, 2928, 2507, 383, 5616, 4302, 3578, 2141, 796, 2059, 5752, 5467, 5322, 2769, 196, 5896, 5692, 5257, 4174, 3521, 3392, 2052, 1710, 5351, 4146, 4082, 2826, 6113, 6112, 6111, 6110, 6109]







    INPUT_FEATURES = 100 
    CLASSES = 9
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("nci60_train_m.txt")
    
    train_text = getIndexData(train_text,index)

    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        if train_classfi[i]=="1" :
            klass = 0
            alldata.addSample(features, klass)
        elif train_classfi[i]=="2" :
            klass = 1
            alldata.addSample(features, klass)
        elif train_classfi[i]=="3" :
            klass = 2
            alldata.addSample(features, klass)
        elif train_classfi[i]=="4" :
            klass = 3
            alldata.addSample(features, klass)
        elif train_classfi[i]=="5" :
            klass = 4
            alldata.addSample(features, klass)
        elif train_classfi[i]=="6" :
            klass = 5
            alldata.addSample(features, klass)
        elif train_classfi[i]=="7" :
            klass = 6
            alldata.addSample(features, klass)
        elif train_classfi[i]=="8" :
            klass = 7
            alldata.addSample(features, klass)
        elif train_classfi[i]=="9" :
            klass = 8
            alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata,'index':index}

def generate_Testdata(index):
    INPUT_FEATURES = 100 
    CLASSES = 9
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("nci60_test_m.txt")
    
    train_text = getIndexData(train_text,index)

    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        if train_classfi[i]=="1" :
            klass = 0
            alldata.addSample(features, klass)
        elif train_classfi[i]=="2" :
            klass = 1
            alldata.addSample(features, klass)
        elif train_classfi[i]=="3" :
            klass = 2
            alldata.addSample(features, klass)
        elif train_classfi[i]=="4" :
            klass = 3
            alldata.addSample(features, klass)
        elif train_classfi[i]=="5" :
            klass = 4
            alldata.addSample(features, klass)
        elif train_classfi[i]=="6" :
            klass = 5
            alldata.addSample(features, klass)
        elif train_classfi[i]=="7" :
            klass = 6
            alldata.addSample(features, klass)
        elif train_classfi[i]=="8" :
            klass = 7
            alldata.addSample(features, klass)
        elif train_classfi[i]=="9" :
            klass = 8
            alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata,'index':index}


def perceptron(hidden_neurons=20, weightdecay=0.01, momentum=0.1):
    INPUT_FEATURES = 100
    CLASSES = 9
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
    #fnn = NetworkReader.readFrom('ncibig(500+83.85).xml')
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
    df.to_excel("ncibigout.xls")
    df = pd.DataFrame(result)
    df.insert(1,'1',tstdata['class'])
    df.to_excel("ncibig.xls")
    error = 0;
    for i in range(len(tstdata['class'])):
        if tstdata['class'][i] != result[i]:
            error = error+1
    #print (len(tstdata['class'])-error)*1.0/len(tstdata['class'])*100
    print AAC(result,tstdata['class'])
    print AUC(np.transpose(tstdata['class'])[0],result.transpose())
    print Fscore(np.transpose(tstdata['class'])[0],result.transpose())
    NetworkWriter.writeToFile(fnn, 'ncibig.xml')
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Add more options if you like
    parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                        default=80,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                        default=0.05,
                        help="weightdecay")
    parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                        default=0.01,
                        help="momentum")
    args = parser.parse_args()

    perceptron(args.hidden_neurons, args.weightdecay, args.momentum)
    # g = generate_data()
    # print g['d']
