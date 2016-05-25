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
        if train_classfi[i]=="A" :
            klass = 0
            train_classfi_number.append(klass)
        elif train_classfi[i]=="C" :
            klass = 1
            train_classfi_number.append(klass)
        elif train_classfi[i]=="N" :
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
    index = [2574, 1512, 3480, 3058, 955, 1261, 6904, 2328, 1115, 2440, 6796, 2428, 6677, 2689, 132, 5817, 5473, 2313, 2290, 3154, 5474, 563, 1220, 6923, 1431, 3000, 1113, 1197, 6900, 2431, 5902, 7120, 1240, 7016, 6475, 5334, 4816, 5908, 3352, 2663, 6741, 1513, 5651, 538, 810, 268, 1733, 6123, 1825, 1546, 2433, 5493, 7011, 3330, 5781, 1087, 3641, 3209, 3776, 1905, 139, 2778, 141, 6060, 685, 6010, 2784, 1106, 5455, 1436, 5544, 217, 6326, 5588, 2777, 390, 1628, 5743, 4083, 5506, 1527, 2712, 3224, 1111, 1453, 3932, 84, 3169, 2884, 1535, 1187, 6047, 2748, 964, 5490, 5881, 4550, 187, 3205, 1295, 1036, 5753, 2278, 6357, 6107, 145, 499, 5461, 2540, 3070, 3135, 4655, 3930, 22, 5460, 3338, 1596, 6043, 341, 1617, 6447, 5806, 2592, 2484, 1482, 7063, 4452, 5633, 5464, 5129, 1568, 2288, 2358, 2727, 2872, 6230, 5548, 5621, 2857, 3927, 4679, 3968, 2752, 2977, 15, 4538, 2725, 5080, 4224, 4147, 5738, 3073, 3305, 3832, 4547, 3395, 3931, 2451, 2353, 6744, 891, 6143, 1225, 1462, 2672, 1427, 1182, 5519, 1496, 1819, 5890, 6601, 81, 1071, 458, 1614, 4680, 1126, 7058, 3088, 990, 6185, 4961, 583, 4669, 5462, 6944, 4829, 1652, 1573, 2041, 1921, 5956, 6879, 2329, 2692, 1207, 6739, 5487, 4586]

    INPUT_FEATURES = 200 
    CLASSES = 3
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("Lung1_train.data")
    
    train_text = getIndexData(train_text,index)

    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        if train_classfi[i]=="A" :
            klass = 0
            alldata.addSample(features, klass)
        elif train_classfi[i]=="C" :
            klass = 1
            alldata.addSample(features, klass)
        elif train_classfi[i]=="N" :
            klass = 2
            alldata.addSample(features, klass)
    return {'minX': 0, 'maxX': 1,
            'minY': 0, 'maxY': 1, 'd': alldata,'index':index}

def generate_Testdata(index):
    INPUT_FEATURES = 200 
    CLASSES = 3
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("Lung1_test.data")

    train_text = getIndexData(train_text,index)

    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    for i in range(len(train_text)):
        features = train_text[i]
        if train_classfi[i]=="A" :
            klass = 0
            alldata.addSample(features, klass)
        elif train_classfi[i]=="C" :
            klass = 1
            alldata.addSample(features, klass)
        elif train_classfi[i]=="N" :
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
    df.to_excel("Lung1out.xls")
    df = pd.DataFrame(result)
    df.insert(1,'1',tstdata['class'])
    df.to_excel("Lung1.xls")
    error = 0;
    for i in range(len(tstdata['class'])):
        if tstdata['class'][i] != result[i]:
            error = error+1
    print AAC(result,tstdata['class'])
    print AUC(np.transpose(tstdata['class'])[0],result.transpose())
    print Fscore(np.transpose(tstdata['class'])[0],result.transpose())
    NetworkWriter.writeToFile(fnn, 'Lung1.xml')
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Add more options if you like
    parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                        default=150,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                        default=0.03,
                        help="weightdecay")
    parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                        default=0.01,
                        help="momentum")
    args = parser.parse_args()

    perceptron(args.hidden_neurons, args.weightdecay, args.momentum)
    # g = generate_data()
    # print g['d']
