import operator
import matplotlib.pyplot as plt
from numpy import *

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m,n = shape(dataMatrix)
	weights = ones(n)

	# Try to find best iteration count
	prevWeights = ones(n)
	diffWeights = zeros(n)
	diffWei = 0.0
	diffWeiMat = []
	prevAreaWei = 1.0
	tmpAreaWei = 1.0

	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4 / (1.0 + j + i) + 0.01
			randIndex = int(random.uniform(0, len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex] * weights))
			error = float(classLabels[randIndex]) - h

			prevWeights = weights
			weights = prevWeights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
		
		diffWeights = (weights - prevWeights)
		diffWeights = diffWeights**2
		diffWei = sum(diffWeights)
		diffWei = diffWei**0.5
		diffWeiMat.append(diffWei)
		
		if j%10 == 0:
			if tmpAreaWei - prevAreaWei > 0.0 and j > (numIter / 5):
				print 'best iteration count : %d' % j
				break

			prevAreaWei = tmpAreaWei
			tmpAreaWei = (sum(diffWeiMat) / len(diffWeiMat))
			print tmpAreaWei
			diffWeiMat = []

	return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def GetMinMaxRanges(dataSet):
	numpyArr = array(dataSet)
	minValue = numpyArr.min(axis=0)
	maxValue = numpyArr.max(axis=0)
	ranges = maxValue - minValue

	return minValue, maxValue, ranges

class LogisticRegression(object):

	def TrainingDataSet(self, purifiedDataSet, labels, attributes):
		return stocGradAscent1(array(purifiedDataSet), labels, 300)

	def TestDataSet(self, origDataSet, trainedDataSet, testDataSet, trainedLabels, testLabels, attributes):
		idx = 0
		errorCount = 0

		totalTest = len(testDataSet)
		currentTest = 0
		targetPercentage = 0.1
		minValue, maxValue, ranges = GetMinMaxRanges(origDataSet)
		weight = trainedDataSet

		for data in testDataSet:
			# normalize
			for i in range(len(data)):
				data[i] = (data[i] - minValue[i]) / (ranges[i])

			value = classifyVector(array(data), weight)
			if int(testLabels[idx]) != int(value):
				errorCount += 1

			currentTest += 1
			if (currentTest/float(totalTest)) >= targetPercentage:
				print '--- %d percent complete' % (targetPercentage * 100)
				targetPercentage += 0.1
			idx += 1

		return errorCount
