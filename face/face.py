from sklearn import datasets
import numpy as np
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import os

# convert a supervised dataset to a classification dataset
def _convert_supervised_to_classification(supervised_dataset,classes):
    classification_dataset = ClassificationDataSet(supervised_dataset.indim,supervised_dataset.outdim,classes)
    
    for n in xrange(0, supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(n)[0], supervised_dataset.getSample(n)[1])

    return classification_dataset

olivetti = datasets.fetch_olivetti_faces()
X, y = olivetti.data, olivetti.target
ds = ClassificationDataSet(4096, 1 , nb_classes=40)
for k in xrange(len(X)): 
	ds.addSample(np.ravel(X[k]),y[k])
tstdata, trndata = ds.splitWithProportion( 0.25 )
tstdata = _convert_supervised_to_classification(tstdata,40)
trndata = _convert_supervised_to_classification(trndata,40)
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )
#fnn = buildNetwork( trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer )
if  os.path.isfile('oliv.xml'): 
 	fnn = NetworkReader.readFrom('oliv.xml')
 	trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
 	trainer.trainEpochs (50)
else:
 	fnn = buildNetwork( trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer )
 	trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
 	trainer.trainEpochs (50)
NetworkWriter.writeToFile(fnn, 'oliv.xml')
print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (dataset=tstdata ), tstdata['class'] )