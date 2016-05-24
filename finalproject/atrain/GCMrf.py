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
    index = [6155, 10566, 11343, 5421, 548, 7677, 830, 7611, 4564, 2269, 864, 1437, 14027, 2211, 3826, 38, 10025, 2407, 2882, 5685, 1051, 14736, 3905, 1072, 6958, 5561, 12830, 3737, 9759, 5520, 13519, 3390, 12198, 6492, 13754, 12847, 9692, 11726, 8663, 8835, 11869, 523, 2709, 5606, 5253, 8762, 8352, 11484, 9139, 2508, 6983, 2740, 1079, 11217, 8214, 6474, 3033, 2699, 877, 14593, 8899, 5684, 1099, 7696, 4562, 12606, 559, 5044, 10294, 3209, 2497, 12424, 8450, 5877, 3143, 13469, 10982, 7691, 6429, 1113, 13636, 13448, 9144, 8932, 2921, 10925, 7283, 12613, 11708, 9811, 7049, 5867, 5124, 4575, 12035, 8867, 15941, 15089, 6699, 5566, 4751, 4229, 7089, 5256, 5205, 5105, 245, 15168, 14052, 11379, 9875, 4816, 4737, 2851, 12981, 8417, 4919, 3229, 2226, 1715, 15489, 5741, 5377, 3506, 2365, 14618, 14084, 12358, 11480, 11289, 11284, 10580, 9763, 8585, 8536, 6435, 6161, 5714, 2935, 1342, 15967, 15919, 13216, 12581, 12549, 12428, 12373, 11994, 11517, 10329, 9354, 8921, 8033, 7033, 4416, 1983, 1846, 15811, 15426, 15225, 15219, 14912, 14858, 14694, 14343, 13960, 12160, 11425, 11095, 10198, 9571, 9423, 8917, 8302, 7715, 7531, 6538, 5098, 4570, 3801, 2577, 2032, 1108, 443, 15948, 15757, 15352, 15169, 15060, 15023, 14664, 14581, 14505, 14451, 14410, 14082, 13882, 13720, 13603, 13500, 13430, 13139, 13082, 12984, 12584, 12051, 11730, 11613, 11319, 11074, 10553, 10211, 9995, 9955, 9936, 9792, 9781, 9733, 9608, 9578, 8951, 6670, 6337, 6228, 5562, 5182, 5176, 4818, 4625, 4619, 4469, 4391, 4294, 4262, 4057, 3519, 3321, 2972, 2690, 2306, 1957, 1754, 1662, 1184, 1160, 604, 16062, 16061, 16060, 16059, 16058, 16057, 16056, 16055, 16054, 16053, 16052, 16051, 16050, 16049, 16048, 16047, 16046, 16045, 16044, 16043, 16042, 16041, 16040, 16039, 16038, 16037, 16036, 16035, 16034, 16033, 16032, 16031, 16030, 16029, 16028, 16027, 16026, 16025, 16024, 16023, 16022, 16021, 16020, 16019, 16018, 16017, 16016, 16015, 16014, 16013, 16012, 16011, 16010, 16009, 16008, 16007, 16006, 16005, 16004, 16003, 16002, 16001, 16000, 15999, 15998, 15997, 15996, 15995, 15994, 15993, 15992, 15991, 15990, 15989, 15988, 15987, 15986, 15985, 15984, 15983, 15982, 15981, 15980, 15979, 15978, 15977, 15976, 15975, 15974, 15973, 15972, 15971, 15970, 15969, 15968, 15966, 15965, 15964, 15963, 15962, 15961, 15960, 15959, 15958, 15957, 15956, 15955, 15954, 15953, 15952, 15951, 15950, 15949, 15947, 15946, 15945, 15944, 15943, 15942, 15940, 15939, 15938, 15937, 15936, 15935, 15934, 15933, 15932, 15931, 15930, 15929, 15928, 15927, 15926, 15925, 15924, 15923, 15922, 15921, 15920, 15918, 15917, 15916, 15915, 15914, 15913, 15912, 15911, 15910, 15909, 15908, 15907, 15906, 15905, 15904, 15903, 15902, 15901, 15900, 15899, 15898, 15897, 15896, 15895, 15894, 15893, 15892, 15891, 15890, 15889, 15888, 15887, 15886, 15885, 15884, 15883, 15882, 15881, 15880, 15879, 15878, 15877, 15876, 15875, 15874, 15873, 15872, 15871, 15870, 15869, 15868, 15867, 15866, 15865, 15864, 15863, 15862, 15861, 15860, 15859, 15858, 15857, 15856, 15855, 15854, 15853, 15852, 15851, 15850, 15849, 15848, 15847, 15846, 15845, 15844, 15843, 15842, 15841, 15840, 15839, 15838, 15837, 15836, 15835, 15834, 15833, 15832, 15831, 15830, 15829, 15828, 15827, 15826, 15825, 15824, 15823, 15822, 15821, 15820, 15819, 15818, 15817, 15816, 15815, 15814, 15813, 15812, 15810, 15809, 15808, 15807, 15806, 15805, 15804, 15803, 15802, 15801, 15800, 15799, 15798, 15797, 15796, 15795, 15794, 15793, 15792, 15791, 15790, 15789, 15788, 15787, 15786, 15785, 15784, 15783, 15782, 15781, 15780, 15779, 15778, 15777, 15776, 15775, 15774, 15773, 15772, 15771, 15770, 15769, 15768, 15767, 15766, 15765, 15764, 15763, 15762, 15761, 15760, 15759, 15758, 15756, 15755, 15754, 15753, 15752, 15751, 15750, 15749, 15748, 15747, 15746, 15745, 15744, 15743, 15742, 15741, 15740, 15739, 15738, 15737, 15736, 15735, 15734, 15733, 15732, 15731, 15730, 15729, 15728, 15727, 15726, 15725, 15724, 15723, 15722, 15721, 15720, 15719, 15718, 15717, 15716, 15715, 15714, 15713, 15712, 15711, 15710, 15709, 15708, 15707, 15706, 15705, 15704, 15703, 15702, 15701, 15700, 15699, 15698, 15697, 15696, 15695, 15694, 15693, 15692, 15691, 15690, 15689, 15688, 15687, 15686, 15685, 15684, 15683, 15682, 15681, 15680, 15679, 15678, 15677, 15676, 15675, 15674, 15673, 15672, 15671, 15670, 15669, 15668, 15667, 15666, 15665, 15664, 15663, 15662, 15661, 15660, 15659, 15658, 15657, 15656, 15655, 15654, 15653, 15652, 15651, 15650, 15649, 15648, 15647, 15646, 15645, 15644, 15643, 15642, 15641, 15640, 15639, 15638, 15637, 15636, 15635, 15634, 15633, 15632, 15631, 15630, 15629, 15628, 15627, 15626, 15625, 15624, 15623, 15622, 15621, 15620, 15619, 15618, 15617, 15616, 15615, 15614, 15613, 15612, 15611, 15610, 15609, 15608, 15607, 15606, 15605, 15604, 15603, 15602, 15601, 15600, 15599, 15598, 15597, 15596, 15595, 15594, 15593, 15592, 15591, 15590, 15589, 15588, 15587, 15586, 15585, 15584, 15583, 15582, 15581, 15580, 15579, 15578, 15577, 15576, 15575, 15574, 15573, 15572, 15571, 15570, 15569, 15568, 15567, 15566, 15565, 15564, 15563, 15562, 15561, 15560, 15559, 15558, 15557, 15556, 15555, 15554, 15553, 15552, 15551, 15550, 15549, 15548, 15547, 15546, 15545, 15544, 15543, 15542, 15541, 15540, 15539, 15538, 15537, 15536, 15535, 15534, 15533, 15532, 15531, 15530, 15529, 15528, 15527, 15526, 15525, 15524, 15523, 15522, 15521, 15520, 15519, 15518, 15517, 15516, 15515, 15514, 15513, 15512, 15511, 15510, 15509, 15508, 15507, 15506, 15505, 15504, 15503, 15502, 15501, 15500, 15499, 15498, 15497, 15496, 15495, 15494, 15493, 15492, 15491, 15490, 15488, 15487, 15486, 15485, 15484, 15483, 15482, 15481, 15480, 15479, 15478, 15477, 15476, 15475, 15474, 15473, 15472, 15471, 15470, 15469, 15468, 15467, 15466, 15465, 15464, 15463, 15462, 15461, 15460, 15459, 15458, 15457, 15456, 15455, 15454, 15453, 15452, 15451, 15450, 15449, 15448, 15447, 15446, 15445, 15444, 15443, 15442, 15441, 15440, 15439, 15438, 15437, 15436, 15435, 15434, 15433, 15432, 15431, 15430, 15429, 15428, 15427, 15425, 15424, 15423, 15422, 15421, 15420, 15419, 15418, 15417, 15416, 15415, 15414, 15413, 15412, 15411, 15410, 15409, 15408, 15407, 15406, 15405, 15404, 15403, 15402, 15401, 15400, 15399, 15398, 15397, 15396, 15395, 15394, 15393, 15392, 15391, 15390, 15389, 15388, 15387, 15386, 15385, 15384, 15383, 15382, 15381, 15380, 15379, 15378, 15377, 15376, 15375, 15374, 15373, 15372, 15371, 15370, 15369, 15368, 15367, 15366, 15365, 15364, 15363, 15362, 15361, 15360, 15359, 15358, 15357, 15356, 15355, 15354, 15353, 15351, 15350, 15349, 15348, 15347, 15346, 15345, 15344, 15343, 15342, 15341, 15340, 15339, 15338, 15337, 15336, 15335, 15334, 15333, 15332, 15331, 15330, 15329, 15328, 15327, 15326, 15325, 15324, 15323, 15322, 15321, 15320, 15319, 15318, 15317, 15316, 15315, 15314, 15313, 15312, 15311, 15310, 15309, 15308, 15307, 15306, 15305, 15304, 15303, 15302, 15301, 15300]


    INPUT_FEATURES = 1000 
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
    INPUT_FEATURES = 1000 
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
    INPUT_FEATURES = 1000
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
