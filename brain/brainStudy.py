# -*- coding: utf-8 -*-

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

# convert a supervised dataset to a classification dataset
def _convert_supervised_to_classification(supervised_dataset,classes):
    classification_dataset = ClassificationDataSet(supervised_dataset.indim,supervised_dataset.outdim,classes)
    
    for n in xrange(0, supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(n)[0], supervised_dataset.getSample(n)[1])

    return classification_dataset


def generate_data():
    INPUT_FEATURES = 3 
    CLASSES = 3
    file_object = open("iris.data.txt")
    lines = file_object.readlines()
    lines.pop()
    for i in xrange(0, len(lines)):
        lines[i] = lines[i].strip().split(",")
    for line in lines:
        for i in xrange(0, len(line)-1):
            line[i] = float(line[i])
    for line in lines:
        if line[4] == 'Iris-setosa':
            line[4] = 0
        elif line[4] == 'Iris-versicolor':
            line[4] = 1
        elif line[4] == 'Iris-virginica':
            line[4] = 2
    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for line in lines:
        features = line[0:3]
        klass = line[4]
        alldata.addSample(features, [klass])
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata}



def perceptron(hidden_neurons=5, weightdecay=0.01, momentum=0.1):
    INPUT_FEATURES = 3
    CLASSES = 3
    HIDDEN_NEURONS = hidden_neurons
    WEIGHTDECAY = weightdecay
    MOMENTUM = momentum
    g = generate_data()
    alldata = g['d']
    tstdata, trndata = alldata.splitWithProportion(0.25)
    trndata = _convert_supervised_to_classification(trndata,CLASSES)
    tstdata = _convert_supervised_to_classification(tstdata,CLASSES)
    trndata._convertToOneOfMany()  
    tstdata._convertToOneOfMany()
    fnn = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,verbose=True, weightdecay=WEIGHTDECAY)
    result = 0;
    for i in range(50):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),trndata['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
        out = fnn.activateOnDataset(tstdata)
        out = out.argmax(axis=1)
        result = out
    df = pd.DataFrame(result)
    df.insert(1,'1',tstdata['class'])
    df.to_excel("see.xls")

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Add more options if you like
    parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                        default=5,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                        default=0.01,
                        help="weightdecay")
    parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                        default=0.1,
                        help="momentum")
    args = parser.parse_args()

    perceptron(args.hidden_neurons, args.weightdecay, args.momentum)
    # g = generate_data()
    # print g['d']
