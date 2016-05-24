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


# convert a supervised dataset to a classification dataset
def _convert_supervised_to_classification(supervised_dataset,classes):
    classification_dataset = ClassificationDataSet(supervised_dataset.indim,supervised_dataset.outdim,classes)
    
    for n in xrange(0, supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(n)[0], supervised_dataset.getSample(n)[1])

    return classification_dataset


def generate_data():
    index = [195,201,2536,485,1586,2720,1176,1689,2953,215,173,282,307,1355,4258,4475,3674,4235,3348,2629,5821,2632,1404,4714,1458,974,3308,23,4957,3941,664,205,1271,5564,5245,991,1300,443,5032,5428,319,1194,3813,2476,2530,1608,4654,5922,4476,5412,1285,3631,2120,79,3825,1362,863,70,2107,1906,1806,475,5478,394,6086,5499,4618,2289,1302,553,6111,4431,4317,2437,1612,5469,1427,369,4617,4450,2709,2370,1813,1264,680,5779,4108,3385,3264,1663,1493,1094,6113,6112,6110,6109,6108,6107,6106,6105,6104,6103,6102,6101,6100,6099,6098,6097,6096,6095,6094,6093,6092,6091,6090,6089,6088,6087,6085,6084,6083,6082,6081,6080,6079,6078,6077,6076,6075,6074,6073,6072,6071,6070,6069,6068,6067,6066,6065,6064,6063,6062,6061,6060,6059,6058,6057,6056,6055,6054,6053,6052,6051,6050,6049,6048,6047,6046,6045,6044,6043,6042,6041,6040,6039,6038,6037,6036,6035,6034,6033,6032,6031,6030,6029,6028,6027,6026,6025,6024,6023,6022,6021,6020,6019,6018,6017,6016,6015,6014,6013,6012,6011,6010,6009,6008,6007,6006,6005,6004]

    INPUT_FEATURES = 200 
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
    INPUT_FEATURES = 200 
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
    INPUT_FEATURES = 200
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
    df.to_excel("ncibigout.xls")
    df = pd.DataFrame(result)
    df.insert(1,'1',tstdata['class'])
    df.to_excel("ncibig.xls")
    error = 0;
    for i in range(len(tstdata['class'])):
        if tstdata['class'][i] != result[i]:
            error = error+1
    print (len(tstdata['class'])-error)*1.0/len(tstdata['class'])*100
    print AAC(result,tstdata['class'])
    NetworkWriter.writeToFile(fnn, 'ncibig.xml')
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
                        default=0.03,
                        help="momentum")
    args = parser.parse_args()

    perceptron(args.hidden_neurons, args.weightdecay, args.momentum)
    # g = generate_data()
    # print g['d']
