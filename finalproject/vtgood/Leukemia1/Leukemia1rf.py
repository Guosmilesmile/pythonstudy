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

def getIndexData(data,index):
    df = pd.DataFrame(data)
    result = []
    for line in index:
        result.append(df[line].values)
    return np.array(result).transpose()


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
        if train_classfi[i]=="B_cell" :
            klass = 0
            train_classfi_number.append(klass)
        elif train_classfi[i]=="AML" :
            klass = 1
            train_classfi_number.append(klass)
        elif train_classfi[i]=="T_cell" :
            klass = 2
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
    index = [629,2641,1009,3280,6224,4041,2994,5553,5299,6509,2014,759,1024,6375,3820,4884,2596,2744,3935,3577,1238,2309,4445,7128,7127,7126,7125,7124,7123,7122,7121,7120,7119,7118,7117,7116,7115,7114,7113,7112,7111,7110,7109,7108,7107,7106,7105,7104,7103,7102,7101,7100,7099,7098,7097,7096,7095,7094,7093,7092,7091,7090,7089,7088,7087,7086,7085,7084,7083,7082,7081,7080,7079,7078,7077,7076,7075,7074,7073,7072,7071,7070,7069,7068,7067,7066,7065,7064,7063,7062,7061,7060,7059,7058,7057,7056,7055,7054,7053,7052,7051,7050,7049,7048,7047,7046,7045,7044,7043,7042,7041,7040,7039,7038,7037,7036,7035,7034,7033,7032,7031,7030,7029,7028,7027,7026,7025,7024,7023,7022,7021,7020,7019,7018,7017,7016,7015,7014,7013,7012,7011,7010,7009,7008,7007,7006,7005,7004,7003,7002,7001,7000,6999,6998,6997,6996,6995,6994,6993,6992,6991,6990,6989,6988,6987,6986,6985,6984,6983,6982,6981,6980,6979,6978,6977,6976,6975,6974,6973,6972,6971,6970,6969,6968,6967,6966,6965,6964,6963,6962,6961,6960,6959,6958,6957,6956,6955,6954,6953,6952]

    INPUT_FEATURES = 200 
    CLASSES = 3
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("Leukemia1_train.data")
    train_text = getIndexData(train_text,index)

    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        if train_classfi[i]=="B_cell" :
            klass = 0
            alldata.addSample(features, klass)
        elif train_classfi[i]=="AML" :
            klass = 1
            alldata.addSample(features, klass)
        elif train_classfi[i]=="T_cell" :
            klass = 2
            alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata,'index':index}

def generate_Testdata(index):
    INPUT_FEATURES = 200 
    CLASSES = 3
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("Leukemia1_test.data")
    
    train_text = getIndexData(train_text,index)


    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        if train_classfi[i]=="B_cell" :
            klass = 0
            alldata.addSample(features, klass)
        elif train_classfi[i]=="AML" :
            klass = 1
            alldata.addSample(features, klass)
        elif train_classfi[i]=="T_cell" :
            klass = 2
            alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata,'index':index}


def perceptron(hidden_neurons=20, weightdecay=0.01, momentum=0.1):
    INPUT_FEATURES = 200
    CLASSES = 3
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
    df.to_excel("Leukemia1out.xls")
    df = pd.DataFrame(result)
    df.insert(1,'1',tstdata['class'])
    df.to_excel("Leukemia1.xls")
    error = 0;
    for i in range(len(tstdata['class'])):
        if tstdata['class'][i] != result[i]:
            error = error+1
    #print (len(tstdata['class'])-error)*1.0/len(tstdata['class'])*100
    print AAC(result,tstdata['class'])
    print AUC(np.transpose(tstdata['class'])[0],result.transpose())
    print Fscore(np.transpose(tstdata['class'])[0],result.transpose())
    NetworkWriter.writeToFile(fnn, 'Leukemia1.xml')
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Add more options if you like
    parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                        default=100,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                        default=0.01,
                        help="weightdecay")
    parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                        default=0.05,
                        help="momentum")
    args = parser.parse_args()

    perceptron(args.hidden_neurons, args.weightdecay, args.momentum)
    # g = generate_data()
    # print g['d']
