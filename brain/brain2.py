from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

# convert a supervised dataset to a classification dataset
def _convert_supervised_to_classification(supervised_dataset):
    classification_dataset = ClassificationDataSet(supervised_dataset.indim,supervised_dataset.outdim,3)
    
    for n in xrange(0, supervised_dataset.getLength()):
        classification_dataset.addSample(supervised_dataset.getSample(n)[0], supervised_dataset.getSample(n)[1])

    return classification_dataset

means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)
for n in xrange(400):
    for klass in range(3):
        input = multivariate_normal(means[klass],cov[klass])
        alldata.addSample(input, [klass])
tstdata1, trndata1 = alldata.splitWithProportion( 0.25 )
tstdata = _convert_supervised_to_classification(tstdata1)
trndata = _convert_supervised_to_classification(trndata1)

tstdata._convertToOneOfMany()
trndata._convertToOneOfMany()

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
ticks = arange(-3.,6.,0.2)
X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=3)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy
for i in range(20):
	trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
	tstresult = percentError( trainer.testOnClassData(dataset=tstdata), tstdata['class'] )
	out = fnn.activateOnDataset(griddata)
	out = out.argmax(axis=1)  # the highest output activation gives the class
	out = out.reshape(X.shape)
	figure(1)
	ioff()  # interactive graphics off
	clf()   # clear the plot
	hold(True) # overplot on
	for c in [0,1,2]:
		here, _ = where(tstdata['class']==c)
		plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
		if out.max()!=out.min():  # safety check against flat field
			contourf(X, Y, out)   # plot the contour
	ion()   # interactive graphics on
	draw()  # update the plot
ioff()
show()
