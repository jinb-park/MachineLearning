import operator
from numpy import *
import matplotlib.pyplot as plt
from math import log

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0

	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec = hstack((reducedFeatVec, featVec[axis+1:]))
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1			# number of attribute
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = 0

	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)	# extract unique value of featList
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)

		infoGain = baseEntropy - newEntropy
		if(infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i

	return bestFeature

def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys(): 
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def createTree(dataSet, attributes):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList): 
		return classList[0]

	if len(dataSet[0]) == 1:
		return majorityCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = attributes[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(attributes[bestFeat])

	featValues = [round(example[bestFeat], 1) for example in dataSet]
	uniqueVals = set(featValues)

	for value in uniqueVals:
		subAttributes = attributes[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subAttributes)

	return myTree

def classifyDecisionTree(inputTree,featLabels,testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	key = testVec[featIndex]
	key = round(key, 1)

	try:
		valueOfFeat = secondDict[key]
	except KeyError:		# underfitting case!!!
		return -1

	if isinstance(valueOfFeat, dict): 
		classLabel = classifyDecisionTree(valueOfFeat, featLabels, testVec)
	else: classLabel = valueOfFeat

	return classLabel

def analyzeDecisionTree(inputTree):
	dnodeCount = 0
	lnodeCount = 0
	maxDepth = 0

	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]

	for key in secondDict.keys():
		if isinstance(secondDict[key], dict):
			dnodeCount += 1

			dCount, lCount, mDepth = analyzeDecisionTree(secondDict[key])
			dnodeCount += dCount
			lnodeCount += lCount

			if mDepth > maxDepth:
				maxDepth = mDepth
		else:
			lnodeCount += 1

	maxDepth += 1
	return dnodeCount, lnodeCount, maxDepth


def GetMinMaxRanges(dataSet):
	numpyArr = array(dataSet)
	minValue = numpyArr.min(axis=0)
	maxValue = numpyArr.max(axis=0)
	ranges = maxValue - minValue

	return minValue, maxValue, ranges

class DecisionTree(object):

	def TrainingDataSet(self, purifiedDataSet, labels, attributes):
		dataSet = []
		idx = 0
		for data in purifiedDataSet:
			newData = zeros(len(data) + 1)

			j = 0
			for j in range(len(data)):
				newData[j] = data[j]
				j += 1
			newData[j] = labels[idx]
			
			dataSet.append( newData )
			idx += 1

		dtree = createTree(dataSet, attributes)
		dnodeCount, lnodeCount, maxDepth = analyzeDecisionTree(dtree)

		print '============== DecisionTree Info ======================='
		print 'DecisionNodeCount : %d' % (dnodeCount)
		print 'LeafNodeCount : %d' % (lnodeCount)
		print 'MaxDepth : %d' % (maxDepth)
		print '========================================================'

		return dtree

	def TestDataSet(self, origDataSet, trainedDataSet, testDataSet, trainedLabels, testLabels, attributes):
		idx = 0
		errorCount = 0
		underfittingCount = 0
		overfittingCount = 0

		totalTest = len(testDataSet)
		currentTest = 0
		targetPercentage = 0.1
		minValue, maxValue, ranges = GetMinMaxRanges(origDataSet)

		for data in testDataSet:
			# normalize
			for i in range(len(data)):
				data[i] = (data[i] - minValue[i]) / (ranges[i])
				data[i] = round(data[i], 1)

			value = classifyDecisionTree(trainedDataSet, attributes, data)

			if float(testLabels[idx]) != float(value):
				errorCount += 1
				if float(value) == float(-1):
					underfittingCount += 1

			currentTest += 1
			if (currentTest/float(totalTest)) >= targetPercentage:
				print '--- %d percent complete' % (targetPercentage * 100)
				targetPercentage += 0.1
			idx += 1

		print '============== DecisionTree Error ======================'
		print 'TotalError : %d' % (errorCount)
		print 'UnderfittingError : %d' % (underfittingCount) 
		print 'OverfittingError : %d' % (errorCount - underfittingCount)  
		print '========================================================'

		return errorCount