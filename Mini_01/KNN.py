import operator
from numpy import *

def classifyKNN(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}  

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

class KNN(object):
	def TrainingDataSet(self, purifiedDataSet):
		return purifiedDataSet

	def TestDataSet(self, trainedDataSet, testDataSet, trainedLabels, testLabels, minValue, maxValue, ranges):
		idx = 0
		errorCount = 0

		totalTest = len(testDataSet)
		currentTest = 0
		targetPercentage = 0.1

		for data in testDataSet:
			# normalize
			for idx in range(len(data)):
				data[idx] = (data[idx] - minValue[idx]) / (ranges[idx])

			value = classifyKNN(data, trainedDataSet, trainedLabels, 3)

			if testLabels[idx] != value:
				errorCount += 1

			currentTest += 1
			if (currentTest/float(totalTest)) >= targetPercentage:
				print '--- %d percent complete' % (targetPercentage * 100)
				targetPercentage += 0.1

		return errorCount
