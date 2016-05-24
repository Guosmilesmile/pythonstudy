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
    index = [3471, 791, 458, 3068, 1542, 524, 278, 526, 5769, 3129, 5440, 166, 4577, 5714, 1692, 546, 402, 2552, 4129, 1894, 4743, 1809, 630, 208, 818, 6034, 3988, 3981, 4580, 134, 1289, 5712, 4723, 4961, 3417, 2630, 994, 689, 5770, 3122, 4823, 4508, 2696, 5566, 2136, 4217, 1503, 1448, 3117, 1161, 5385, 6095, 2197, 325, 2310, 4990, 2009, 5880, 3900, 1715, 1573, 1488, 1125, 3533, 3004, 55, 4424, 3077, 499, 144, 5976, 4643, 3219, 2328, 1770, 1510, 770, 107, 1625, 4684, 4544, 4470, 3684, 3607, 942, 671, 5796, 3773, 2204, 2083, 345, 3942, 6113, 6112, 6111, 6110, 6109, 6108, 6107, 6106, 6105, 6104, 6103, 6102, 6101, 6100, 6099, 6098, 6097, 6096, 6094, 6093, 6092, 6091, 6090, 6089, 6088, 6087, 6086, 6085, 6084, 6083, 6082, 6081, 6080, 6079, 6078, 6077, 6076, 6075, 6074, 6073, 6072, 6071, 6070, 6069, 6068, 6067, 6066, 6065, 6064, 6063, 6062, 6061, 6060, 6059, 6058, 6057, 6056, 6055, 6054, 6053, 6052, 6051, 6050, 6049, 6048, 6047, 6046, 6045, 6044, 6043, 6042, 6041, 6040, 6039, 6038, 6037, 6036, 6035, 6033, 6032, 6031, 6030, 6029, 6028, 6027, 6026, 6025, 6024, 6023, 6022, 6021, 6020, 6019, 6018, 6017, 6016, 6015, 6014, 6013, 6012, 6011, 6010, 6009, 6008, 6007, 6006, 6005, 6004, 6003, 6002, 6001, 6000, 5999, 5998, 5997, 5996, 5995, 5994, 5993, 5992, 5991, 5990, 5989, 5988, 5987, 5986, 5985, 5984, 5983, 5982, 5981, 5980, 5979, 5978, 5977, 5975, 5974, 5973, 5972, 5971, 5970, 5969, 5968, 5967, 5966, 5965, 5964, 5963, 5962, 5961, 5960, 5959, 5958, 5957, 5956, 5955, 5954, 5953, 5952, 5951, 5950, 5949, 5948, 5947, 5946, 5945, 5944, 5943, 5942, 5941, 5940, 5939, 5938, 5937, 5936, 5935, 5934, 5933, 5932, 5931, 5930, 5929, 5928, 5927, 5926, 5925, 5924, 5923, 5922, 5921, 5920, 5919, 5918, 5917, 5916, 5915, 5914, 5913, 5912, 5911, 5910, 5909, 5908, 5907, 5906, 5905, 5904, 5903, 5902, 5901, 5900, 5899, 5898, 5897, 5896, 5895, 5894, 5893, 5892, 5891, 5890, 5889, 5888, 5887, 5886, 5885, 5884, 5883, 5882, 5881, 5879, 5878, 5877, 5876, 5875, 5874, 5873, 5872, 5871, 5870, 5869, 5868, 5867, 5866, 5865, 5864, 5863, 5862, 5861, 5860, 5859, 5858, 5857, 5856, 5855, 5854, 5853, 5852, 5851, 5850, 5849, 5848, 5847, 5846, 5845, 5844, 5843, 5842, 5841, 5840, 5839, 5838, 5837, 5836, 5835, 5834, 5833, 5832, 5831, 5830, 5829, 5828, 5827, 5826, 5825, 5824, 5823, 5822, 5821, 5820, 5819, 5818, 5817, 5816, 5815, 5814, 5813, 5812, 5811, 5810, 5809, 5808, 5807, 5806, 5805, 5804, 5803, 5802, 5801, 5800, 5799, 5798, 5797, 5795, 5794, 5793, 5792, 5791, 5790, 5789, 5788, 5787, 5786, 5785, 5784, 5783, 5782, 5781, 5780, 5779, 5778, 5777, 5776, 5775, 5774, 5773, 5772, 5771, 5768, 5767, 5766, 5765, 5764, 5763, 5762, 5761, 5760, 5759, 5758, 5757, 5756, 5755, 5754, 5753, 5752, 5751, 5750, 5749, 5748, 5747, 5746, 5745, 5744, 5743, 5742, 5741, 5740, 5739, 5738, 5737, 5736, 5735, 5734, 5733, 5732, 5731, 5730, 5729, 5728, 5727, 5726, 5725, 5724, 5723, 5722, 5721, 5720, 5719, 5718, 5717, 5716, 5715, 5713, 5711, 5710, 5709, 5708, 5707, 5706, 5705, 5704, 5703, 5702, 5701, 5700, 5699, 5698, 5697]






    INPUT_FEATURES = 500 
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
    INPUT_FEATURES = 500 
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
    INPUT_FEATURES = 500
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
    fnn = NetworkReader.readFrom('ncibig(500+83.85).xml')
    #fnn = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,outclass=SoftmaxLayer)
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
