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
    index = [201, 2080, 230, 1553, 195, 2157, 2544, 5390, 971, 990, 2009, 3179, 403, 4733, 166, 1042, 3330, 43, 1582, 3965, 1093, 282, 3535, 803, 1912, 64, 3333, 2340, 3269, 5235, 5279, 4642, 2032, 1891, 3661, 2018, 417, 332, 3948, 820, 2847, 4202, 3588, 3268, 550, 2354, 1603, 734, 4209, 5314, 3944, 3638, 1150, 4843, 1217, 5438, 4366, 1386, 3655, 1548, 1523, 2496, 2921, 2460, 2111, 1436, 863, 446, 5913, 4731, 2105, 1672, 326, 4937, 1814, 1189, 226, 4659, 4071, 3082, 1694, 980, 321, 6009, 5386, 5265, 5073, 4922, 4796, 3359, 3284, 2611, 1727, 1495, 733, 457, 4009, 2224, 276, 6113, 6112, 6111, 6110, 6109, 6108, 6107, 6106, 6105, 6104, 6103, 6102, 6101, 6100, 6099, 6098, 6097, 6096, 6095, 6094, 6093, 6092, 6091, 6090, 6089, 6088, 6087, 6086, 6085, 6084, 6083, 6082, 6081, 6080, 6079, 6078, 6077, 6076, 6075, 6074, 6073, 6072, 6071, 6070, 6069, 6068, 6067, 6066, 6065, 6064, 6063, 6062, 6061, 6060, 6059, 6058, 6057, 6056, 6055, 6054, 6053, 6052, 6051, 6050, 6049, 6048, 6047, 6046, 6045, 6044, 6043, 6042, 6041, 6040, 6039, 6038, 6037, 6036, 6035, 6034, 6033, 6032, 6031, 6030, 6029, 6028, 6027, 6026, 6025, 6024, 6023, 6022, 6021, 6020, 6019, 6018, 6017, 6016, 6015, 6014, 6013, 6012, 6011, 6010, 6008, 6007, 6006, 6005, 6004, 6003, 6002, 6001, 6000, 5999, 5998, 5997, 5996, 5995, 5994, 5993, 5992, 5991, 5990, 5989, 5988, 5987, 5986, 5985, 5984, 5983, 5982, 5981, 5980, 5979, 5978, 5977, 5976, 5975, 5974, 5973, 5972, 5971, 5970, 5969, 5968, 5967, 5966, 5965, 5964, 5963, 5962, 5961, 5960, 5959, 5958, 5957, 5956, 5955, 5954, 5953, 5952, 5951, 5950, 5949, 5948, 5947, 5946, 5945, 5944, 5943, 5942, 5941, 5940, 5939, 5938, 5937, 5936, 5935, 5934, 5933, 5932, 5931, 5930, 5929, 5928, 5927, 5926, 5925, 5924, 5923, 5922, 5921, 5920, 5919, 5918, 5917, 5916, 5915, 5914, 5912, 5911, 5910, 5909, 5908, 5907, 5906, 5905, 5904, 5903, 5902, 5901, 5900, 5899, 5898, 5897, 5896, 5895, 5894, 5893, 5892, 5891, 5890, 5889, 5888, 5887, 5886, 5885, 5884, 5883, 5882, 5881, 5880, 5879, 5878, 5877, 5876, 5875, 5874, 5873, 5872, 5871, 5870, 5869, 5868, 5867, 5866, 5865, 5864, 5863, 5862, 5861, 5860, 5859, 5858, 5857, 5856, 5855, 5854, 5853, 5852, 5851, 5850, 5849, 5848, 5847, 5846, 5845, 5844, 5843, 5842, 5841, 5840, 5839, 5838, 5837, 5836, 5835, 5834, 5833, 5832, 5831, 5830, 5829, 5828, 5827, 5826, 5825, 5824, 5823, 5822, 5821, 5820, 5819, 5818, 5817, 5816, 5815, 5814, 5813, 5812, 5811, 5810, 5809, 5808, 5807, 5806, 5805, 5804, 5803, 5802, 5801, 5800, 5799, 5798, 5797, 5796, 5795, 5794, 5793, 5792, 5791, 5790, 5789, 5788, 5787, 5786, 5785, 5784, 5783, 5782, 5781, 5780, 5779, 5778, 5777, 5776, 5775, 5774, 5773, 5772, 5771, 5770, 5769, 5768, 5767, 5766, 5765, 5764, 5763, 5762, 5761, 5760, 5759, 5758, 5757, 5756, 5755, 5754, 5753, 5752, 5751, 5750, 5749, 5748, 5747, 5746, 5745, 5744, 5743, 5742, 5741, 5740, 5739, 5738, 5737, 5736, 5735, 5734, 5733, 5732, 5731, 5730, 5729, 5728, 5727, 5726, 5725, 5724, 5723, 5722, 5721, 5720, 5719, 5718, 5717, 5716, 5715, 5714, 5713, 5712, 5711, 5710, 5709, 5708, 5707, 5706, 5705, 5704, 5703, 5702, 5701, 5700, 5699, 5698, 5697, 5696, 5695, 5694, 5693, 5692, 5691, 5690, 5689, 5688, 5687, 5686, 5685, 5684, 5683, 5682, 5681, 5680, 5679, 5678, 5677, 5676, 5675, 5674, 5673, 5672, 5671, 5670, 5669, 5668, 5667, 5666, 5665, 5664, 5663, 5662, 5661, 5660, 5659, 5658, 5657, 5656, 5655, 5654, 5653, 5652, 5651, 5650, 5649, 5648, 5647, 5646, 5645, 5644, 5643, 5642, 5641, 5640, 5639, 5638, 5637, 5636, 5635, 5634, 5633, 5632, 5631, 5630, 5629, 5628, 5627, 5626, 5625, 5624, 5623, 5622, 5621, 5620, 5619, 5618, 5617, 5616, 5615, 5614, 5613, 5612, 5611, 5610, 5609, 5608, 5607, 5606, 5605, 5604, 5603, 5602, 5601, 5600, 5599, 5598, 5597, 5596, 5595, 5594, 5593, 5592, 5591, 5590, 5589, 5588, 5587, 5586, 5585, 5584, 5583, 5582, 5581, 5580, 5579, 5578, 5577, 5576, 5575, 5574, 5573, 5572, 5571, 5570, 5569, 5568, 5567, 5566, 5565, 5564, 5563, 5562, 5561, 5560, 5559, 5558, 5557, 5556, 5555, 5554, 5553, 5552, 5551, 5550, 5549, 5548, 5547, 5546, 5545, 5544, 5543, 5542, 5541, 5540, 5539, 5538, 5537, 5536, 5535, 5534, 5533, 5532, 5531, 5530, 5529, 5528, 5527, 5526, 5525, 5524, 5523, 5522, 5521, 5520, 5519, 5518, 5517, 5516, 5515, 5514, 5513, 5512, 5511, 5510, 5509, 5508, 5507, 5506, 5505, 5504, 5503, 5502, 5501, 5500, 5499, 5498, 5497, 5496, 5495, 5494, 5493, 5492, 5491, 5490, 5489, 5488, 5487, 5486, 5485, 5484, 5483, 5482, 5481, 5480, 5479, 5478, 5477, 5476, 5475, 5474, 5473, 5472, 5471, 5470, 5469, 5468, 5467, 5466, 5465, 5464, 5463, 5462, 5461, 5460, 5459, 5458, 5457, 5456, 5455, 5454, 5453, 5452, 5451, 5450, 5449, 5448, 5447, 5446, 5445, 5444, 5443, 5442, 5441, 5440, 5439, 5437, 5436, 5435, 5434, 5433, 5432, 5431, 5430, 5429, 5428, 5427, 5426, 5425, 5424, 5423, 5422, 5421, 5420, 5419, 5418, 5417, 5416, 5415, 5414, 5413, 5412, 5411, 5410, 5409, 5408, 5407, 5406, 5405, 5404, 5403, 5402, 5401, 5400, 5399, 5398, 5397, 5396, 5395, 5394, 5393, 5392, 5391, 5389, 5388, 5387, 5385, 5384, 5383, 5382, 5381, 5380, 5379, 5378, 5377, 5376, 5375, 5374, 5373, 5372, 5371, 5370, 5369, 5368, 5367, 5366, 5365, 5364, 5363, 5362, 5361, 5360, 5359, 5358, 5357, 5356, 5355, 5354, 5353, 5352, 5351, 5350, 5349, 5348, 5347, 5346, 5345, 5344, 5343, 5342, 5341, 5340, 5339, 5338, 5337, 5336, 5335, 5334, 5333, 5332, 5331, 5330, 5329, 5328, 5327, 5326, 5325, 5324, 5323, 5322, 5321, 5320, 5319, 5318, 5317, 5316, 5315, 5313, 5312, 5311, 5310, 5309, 5308, 5307, 5306, 5305, 5304, 5303, 5302, 5301, 5300, 5299, 5298, 5297, 5296, 5295, 5294, 5293, 5292, 5291, 5290, 5289, 5288, 5287, 5286, 5285, 5284, 5283, 5282, 5281, 5280, 5278, 5277, 5276, 5275, 5274, 5273, 5272, 5271, 5270, 5269, 5268, 5267, 5266, 5264, 5263, 5262, 5261, 5260, 5259, 5258, 5257, 5256, 5255, 5254, 5253, 5252, 5251, 5250, 5249, 5248, 5247, 5246, 5245, 5244, 5243, 5242, 5241, 5240, 5239, 5238, 5237, 5236, 5234, 5233, 5232, 5231, 5230, 5229, 5228, 5227, 5226, 5225, 5224, 5223, 5222, 5221, 5220, 5219, 5218, 5217, 5216, 5215, 5214, 5213, 5212, 5211, 5210, 5209, 5208, 5207, 5206, 5205, 5204]




    INPUT_FEATURES = 16063 
    CLASSES = 15
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("GCM_train.data")
    
    #train_text = getIndexData(train_text,index)

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
    INPUT_FEATURES = 16063 
    CLASSES = 15
    #train_text,train_classfi = getTargetData("Breast_train.data")

    #Load boston housing dataset as an example
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData("GCM_test.data")
    #train_text = getIndexData(train_text,index)
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
    INPUT_FEATURES = 16063
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
    #fnn = NetworkReader.readFrom('GCM(500+52.31).xml')
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,verbose=True, weightdecay=WEIGHTDECAY,learningrate=0.01)
    result = 0;
    ssss = 0;
    for i in range(100):
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
