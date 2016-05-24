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
    index = [6308, 2407, 9898, 7611, 12970, 5652, 14007, 9880, 548, 4014, 4897, 13790, 7677, 9437, 1275, 5561, 10984, 557, 9737, 6166, 6769, 1942, 1825, 153, 1847, 5817, 2235, 122, 5771, 6087, 15287, 8612, 4335, 412, 10163, 1683, 8755, 10431, 6677, 15806, 295, 297, 6549, 661, 6903, 5445, 3809, 9720, 1174, 1625, 6773, 8835, 8215, 6247, 1156, 8518, 7024, 4307, 6252, 12282, 3251, 984, 13671, 14557, 11652, 9592, 6003, 9856, 8207, 13285, 9770, 9645, 6729, 12453, 12013, 7046, 8395, 5995, 2012, 9050, 8649, 5965, 5448, 258, 11313, 7694, 7132, 4579, 4264, 2805, 13888, 14339, 12373, 11491, 11179, 7439, 5496, 1055, 13967, 12596, 7358, 4664, 2787, 205, 6053, 4119, 1718, 441, 13884, 12135, 13314, 12692, 12686, 9751, 8371, 5931, 4418, 1218, 637, 36, 15313, 11159, 10954, 10123, 8961, 7526, 7140, 4265, 3362, 2165, 2060, 351, 15666, 15120, 14898, 13703, 9479, 9095, 8519, 6373, 5401, 4301, 4206, 3604, 13154, 11949, 11160, 9438, 9056, 8325, 8231, 8091, 6970, 6669, 6134, 5167, 2087, 593, 39, 29, 15919, 15367, 14436, 14188, 14108, 13988, 12887, 12826, 12162, 12046, 11867, 9234, 9020, 8704, 7684, 7387, 7289, 7163, 6845, 6828, 2687, 2555, 1405, 807, 186, 16010, 15904, 15701, 15647, 15614, 15532, 15391, 15364, 15215, 15166, 14970, 14249, 14231, 14187, 14177, 14065, 13687, 13595, 13546, 13483, 13179, 12513, 11982, 11959, 11922, 11660, 10930, 10923, 10842, 10201, 9448, 9380, 9376, 9084, 8966, 8658, 8634, 8462, 8157, 8119, 8117, 7678, 7600, 7308, 7197, 7094, 6886, 6258, 6255, 6209, 5904, 5777, 5423, 4699, 4582, 4472, 3274, 3128, 3032, 2824, 2084, 1591, 1535, 1401, 1397, 1320, 1188, 1061, 1019, 972, 813, 365, 313, 235, 72, 20, 16062, 16061, 16060, 16059, 16058, 16057, 16056, 16055, 16054, 16053, 16052, 16051, 16050, 16049, 16048, 16047, 16046, 16045, 16044, 16043, 16042, 16041, 16040, 16039, 16038, 16037, 16036, 16035, 16034, 16033, 16032, 16031, 16030, 16029, 16028, 16027, 16026, 16025, 16024, 16023, 16022, 16021, 16020, 16019, 16018, 16017, 16016, 16015, 16014, 16013, 16012, 16011, 16009, 16008, 16007, 16006, 16005, 16004, 16003, 16002, 16001, 16000, 15999, 15998, 15997, 15996, 15995, 15994, 15993, 15992, 15991, 15990, 15989, 15988, 15987, 15986, 15985, 15984, 15983, 15982, 15981, 15980, 15979, 15978, 15977, 15976, 15975, 15974, 15973, 15972, 15971, 15970, 15969, 15968, 15967, 15966, 15965, 15964, 15963, 15962, 15961, 15960, 15959, 15958, 15957, 15956, 15955, 15954, 15953, 15952, 15951, 15950, 15949, 15948, 15947, 15946, 15945, 15944, 15943, 15942, 15941, 15940, 15939, 15938, 15937, 15936, 15935, 15934, 15933, 15932, 15931, 15930, 15929, 15928, 15927, 15926, 15925, 15924, 15923, 15922, 15921, 15920, 15918, 15917, 15916, 15915, 15914, 15913, 15912, 15911, 15910, 15909, 15908, 15907, 15906, 15905, 15903, 15902, 15901, 15900, 15899, 15898, 15897, 15896, 15895, 15894, 15893, 15892, 15891, 15890, 15889, 15888, 15887, 15886, 15885, 15884, 15883, 15882, 15881, 15880, 15879, 15878, 15877, 15876, 15875, 15874, 15873, 15872, 15871, 15870, 15869, 15868, 15867, 15866, 15865, 15864, 15863, 15862, 15861, 15860, 15859, 15858, 15857, 15856, 15855, 15854, 15853, 15852, 15851, 15850, 15849, 15848, 15847, 15846, 15845, 15844, 15843, 15842, 15841, 15840, 15839, 15838, 15837, 15836, 15835, 15834, 15833, 15832, 15831, 15830, 15829, 15828, 15827, 15826, 15825, 15824, 15823, 15822, 15821]


    INPUT_FEATURES = 500 
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
    INPUT_FEATURES = 500 
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
    INPUT_FEATURES = 500
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
    fnn = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,verbose=True, weightdecay=WEIGHTDECAY,learningrate=0.01)
    result = 0;
    ssss = 0;
    for i in range(150):
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
    print AAC(result,tstdata['class'])
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
