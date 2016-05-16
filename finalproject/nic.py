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
        all_text.append(line.strip().split(" "))
    train_classfi  = all_text[0]
    for i in range(len(all_text)):
        if(i!=0):
            train_text.append(all_text[i])
    train_text = np.array(train_text)
    train_classfi = np.array(train_classfi)
    return train_text.transpose(),train_classfi


# convert a supervised dataset to a classification dataset
def _convert_supervised_to_classification(supervised_dataset,classes):
    classification_dataset = ClassificationDataSet(supervised_dataset.indim,supervised_dataset.outdim,classes)
    
    for n in xrange(0, supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(n)[0], supervised_dataset.getSample(n)[1])

    return classification_dataset

def generate_data():
    INPUT_FEATURES = 6114 
    CLASSES = 10

    train_text,train_classfi = getTargetData("nci60_train_m.txt")
    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        klass = train_classfi[i]
        alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata}

def generate_Testdata():
    INPUT_FEATURES = 6114 
    CLASSES = 10

    train_text,train_classfi = getTargetData("nci60_test_m.txt")

    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        klass = train_classfi[i]
        alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata}


def perceptron(hidden_neurons=20, weightdecay=0.01, momentum=0.1):
    INPUT_FEATURES = 6114
    CLASSES = 10
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
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,verbose=True, weightdecay=WEIGHTDECAY,learningrate=0.004)
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
    df.to_excel("nciout.xls")
    df = pd.DataFrame(result)
    df.insert(1,'1',tstdata['class'])
    df.to_excel("nci.xls")
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
                        default=200,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                        default=0.05,
                        help="weightdecay")
    parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                        default=0.1,
                        help="momentum")
    args = parser.parse_args()

    perceptron(args.hidden_neurons, args.weightdecay, args.momentum)
    
    #g = generate_data()
    # print len(g['d'])
