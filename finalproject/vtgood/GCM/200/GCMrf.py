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
        if train_classfi[i]=="Breast" :
            klass = 0
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Prostate" :
            klass = 1
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Lung" :
            klass = 2
            train_classfi_number.append(klass)
        elif train_classfi[i]=="normal" :
            klass = 3
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Lymphoma" :
            klass = 4
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Bladder" :
            klass = 5
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Melanoma" :
            klass = 6
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Uterus" :
            klass = 7
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Leukemia" :
            klass = 8
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Renal" :
            klass = 9
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Pancreas" :
            klass = 10
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Ovary" :
            klass = 11
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Mesothelioma" :
            klass = 12
            train_classfi_number.append(klass)
        elif train_classfi[i]=="CNS" :
            klass = 13
            train_classfi_number.append(klass)
        elif train_classfi[i]=="Colorectal" :
            klass = 14
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


# convert a supervised dataset to a classification dataset
def _convert_supervised_to_classification(supervised_dataset,classes):
    classification_dataset = ClassificationDataSet(supervised_dataset.indim,supervised_dataset.outdim,classes)
    
    for n in xrange(0, supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(n)[0], supervised_dataset.getSample(n)[1])

    return classification_dataset


def generate_data():
    index = [9154,5123,2407,680,548,8016,15755,9861,461,5552,6834,6268,14112,15285,13065,8838,2962,6581,4025,14928,10521,1413,3587,3537,13462,9809,4128,15806,4884,2084,7818,8294,12308,8789,5328,5817,7663,6299,15295,3547,1673,5940,6085,6368,6006,5520,14228,8608,7822,3237,10927,12268,2852,6903,13001,10775,4852,14487,10885,14948,15239,8787,6886,15720,13436,4102,7832,5071,11062,15004,14888,12560,4381,14283,6892,14753,10132,6937,2393,465,11791,8533,2174,6739,4316,251,11438,10288,6658,6439,6711,5173,11590,1452,524,15677,13742,11881,9299,7499,7068,11457,11128,4936,1634,14692,13352,11896,11895,11494,9704,6878,10112,10027,10207,6946,6604,5563,3590,2817,2661,9667,9609,8368,7538,6830,1909,1385,15043,14006,11050,10743,10306,9574,9546,9267,9232,8546,8452,8027,7465,5453,1903,1747,1367,15496,14231,13894,12340,11433,11118,9223,8369,8017,7324,6737,5047,4635,4631,3685,3418,3215,1395,835,690,15808,15210,13829,13798,13303,13220,13078,12416,12407,12082,11940,11266,9794,9643,8825,8600,8446,7892,6972,6728,6559,5759,5091,4640,4209,3214,1994,1599,1447,1082,15881,15810,15586,15564,15150]



    INPUT_FEATURES = 200 
    CLASSES = 15
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("GCM_train.data")
    
    train_text = getIndexData(train_text,index)

    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        if train_classfi[i]=="Breast" :
            klass = 0
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Prostate" :
            klass = 1
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Lung" :
            klass = 2
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Colorectal" :
            klass = 3
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Lymphoma" :
            klass = 4
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Bladder" :
            klass = 5
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Melanoma" :
            klass = 6
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Uterus" :
            klass = 7
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Leukemia" :
            klass = 8
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Renal" :
            klass = 9
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Pancreas" :
            klass = 10
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Ovary" :
            klass = 11
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Mesothelioma" :
            klass = 12
            alldata.addSample(features, klass)
        elif train_classfi[i]=="CNS" :
            klass = 13
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Colorectal" :
            klass = 14
            alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata,'index':index}

def generate_Testdata(index):
    INPUT_FEATURES = 200 
    CLASSES = 15
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("GCM_test.data")
    train_text = getIndexData(train_text,index)
    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        if train_classfi[i]=="Breast" :
            klass = 0
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Prostate" :
            klass = 1
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Lung" :
            klass = 2
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Colorectal" :
            klass = 3
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Lymphoma" :
            klass = 4
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Bladder" :
            klass = 5
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Melanoma" :
            klass = 6
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Uterus" :
            klass = 7
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Leukemia" :
            klass = 8
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Renal" :
            klass = 9
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Pancreas" :
            klass = 10
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Ovary" :
            klass = 11
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Mesothelioma" :
            klass = 12
            alldata.addSample(features, klass)
        elif train_classfi[i]=="CNS" :
            klass = 13
            alldata.addSample(features, klass)
        elif train_classfi[i]=="Colorectal" :
            klass = 14
            alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata,'index':index}


def perceptron(hidden_neurons=20, weightdecay=0.01, momentum=0.1):
    INPUT_FEATURES = 200
    CLASSES = 15
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
    #fnn = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,outclass=SoftmaxLayer)
    fnn = NetworkReader.readFrom('GCM(200+70.87).xml')
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,verbose=True, weightdecay=WEIGHTDECAY,learningrate=0.01)
    result = 0;
    ssss = 0;
    for i in range(1):
        #trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),trndata['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
        out = fnn.activateOnDataset(tstdata)
        ssss = out
        out = out.argmax(axis=1)
        result = out
    df = pd.DataFrame(ssss)
    df.to_excel("GCMout.xls")
    df = pd.DataFrame(result)
    df.insert(1,'1',tstdata['class'])
    df.to_excel("GCM.xls")
    error = 0;
    for i in range(len(tstdata['class'])):
        if tstdata['class'][i] != result[i]:
            error = error+1
    #print (len(tstdata['class'])-error)*1.0/len(tstdata['class'])*100
    print AAC(result,tstdata['class'])
    print AUC(np.transpose(tstdata['class'])[0],result.transpose())
    print Fscore(np.transpose(tstdata['class'])[0],result.transpose())
    NetworkWriter.writeToFile(fnn, 'GCM.xml')
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Add more options if you like
    parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                        default=200,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                        default=0.008,
                        help="weightdecay")
    parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                        default=0.01,
                        help="momentum")
    args = parser.parse_args()

    perceptron(args.hidden_neurons, args.weightdecay, args.momentum)
    # g = generate_data()
    # print g['d']
