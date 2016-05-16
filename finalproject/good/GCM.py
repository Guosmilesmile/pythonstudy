# -*- coding: utf-8 -*-

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
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
    return train_text.transpose(),train_classfi


# convert a supervised dataset to a classification dataset
def _convert_supervised_to_classification(supervised_dataset,classes):
    classification_dataset = ClassificationDataSet(supervised_dataset.indim,supervised_dataset.outdim,classes)
    
    for n in xrange(0, supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(n)[0], supervised_dataset.getSample(n)[1])

    return classification_dataset

def generate_data():
    INPUT_FEATURES = 16063 
    CLASSES = 15

    train_text,train_classfi = getTargetData("GCM_train.data")

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
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata}

def generate_Testdata():
    INPUT_FEATURES = 16063 
    CLASSES = 5

    train_text,train_classfi = getTargetData("GCM_test.data")

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
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata}


def perceptron(hidden_neurons=20, weightdecay=0.01, momentum=0.1):
    INPUT_FEATURES = 16063
    CLASSES = 15
    HIDDEN_NEURONS = hidden_neurons
    WEIGHTDECAY = weightdecay
    MOMENTUM = momentum
    
    g = generate_data()
    alldata = g['d']
    testdata = generate_Testdata()['d']
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
    for i in range(50):
        trainer.trainEpochs(1)
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
    print (len(tstdata['class'])-error)*1.0/len(tstdata['class'])*100
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Add more options if you like
    parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                        default=100,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                        default=0.02,
                        help="weightdecay")
    parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                        default=0.1,
                        help="momentum")
    args = parser.parse_args()

    perceptron(args.hidden_neurons, args.weightdecay, args.momentum)
    
    # g = generate_data()
    # print len(g['d'])
