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

def AUC(listanswer,listtest):
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
        total = total + float(tp)/(tp+fn)*100+float(tn)/(fp+tn)*100
    return total/len(labels)/2


def Fscore(listanswer,listtest):
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
                    fn = tn + 1
                else:
                    tn = fn + 1
        pre1 = pre1 + tp
        pre2 = pre2 + tp + fp
        rec = rec + tp + fn
    pre=float(pre1)/pre2
    rec=float(pre1)/rec
    return 2*pre*rec/(pre+rec)*100

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


def generate_data():
    index = [8673,1646,116,2191,4326,6718,7796,8531,8763,5646,3626,5451,2004,8079,4044,6471,675,3746,6338,3149,4880,4869,6213,5316,3544,1046,7739,8309,4147,5526,5555,1504,1625,2680,5814,1305,3998,794,4355,6788,3343,867,343,3706,6902,4250,9014,5478,788,5323,677,9215,9214,9213,9212,9211,9210,9209,9208,9207,9206,9205,9204,9203,9202,9201,9200,9199,9198,9197,9196,9195,9194,9193,9192,9191,9190,9189,9188,9187,9186,9185,9184,9183,9182,9181,9180,9179,9178,9177,9176,9175,9174,9173,9172,9171,9170,9169,9168,9167,9166,9165,9164,9163,9162,9161,9160,9159,9158,9157,9156,9155,9154,9153,9152,9151,9150,9149,9148,9147,9146,9145,9144,9143,9142,9141,9140,9139,9138,9137,9136,9135,9134,9133,9132,9131,9130,9129,9128,9127,9126,9125,9124,9123,9122,9121,9120,9119,9118,9117,9116,9115,9114,9113,9112,9111,9110,9109,9108,9107,9106,9105,9104,9103,9102,9101,9100,9099,9098,9097,9096,9095,9094,9093,9092,9091,9090,9089,9088,9087,9086,9085,9084,9083,9082,9081,9080,9079,9078,9077,9076,9075,9074,9073,9072,9071,9070,9069,9068,9067]

    INPUT_FEATURES = 200 
    CLASSES = 5
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("Breast_train.data")

    train_text = getIndexData(train_text,index)    

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
            'minY': 0, 'maxY': 1, 'd': alldata,'index':index}

def generate_Testdata(index):
    INPUT_FEATURES = 200 
    CLASSES = 5
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("Breast_test.data")
    
    train_text = getIndexData(train_text,index)   

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
            'minY': 0, 'maxY': 1, 'd': alldata,'index':index}


def perceptron(hidden_neurons=20, weightdecay=0.01, momentum=0.1):
    INPUT_FEATURES = 200
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
    for i in range(200):
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
    print AUC(result,tstdata['class'])
    print Fscore(result,tstdata['class'])
    NetworkWriter.writeToFile(fnn, 'breast.xml')

def getIndexData(data,index):
    df = pd.DataFrame(data)
    result = []
    for line in index:
        result.append(df[line].values)
    return np.array(result).transpose()

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Add more options if you like
    parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                        default=100,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                        default=0.04,
                        help="weightdecay")
    parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                        default=0.01,
                        help="momentum")
    args = parser.parse_args()

 
    perceptron(args.hidden_neurons, args.weightdecay, args.momentum)
    # g = generate_data()
    # print g['d']
