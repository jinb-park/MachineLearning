import operator
import sys
import time
from KNN import KNN
from AdultDataSet import AdultDataSet
from numpy import *

def GetClassifier(algoName):
	if algoName == 'KNN':
		return KNN()
	elif algoName == 'DecisionTree':
		return DecisionTree()
	else:
		print 'Not Support Algorithm'
		exit()

def GetDataSetBuilder(dataFileName):
	if dataFileName.find('adult') > -1:
		return AdultDataSet()
	else:
		return AdultDataSet()

def DoTrainingAndTest(dataFileName, testFileName, algoName):
	print 'DataFileName : %s, TestFileName : %s, AlgorithmName : %s' % (dataFileName, testFileName, algoName)
	errorCount = 0

	# 
	classifier = GetClassifier(algoName)
	dataSetBuilder = GetDataSetBuilder(dataFileName)

	#
	quantitative = False
	if type(classifier).__name__ == 'DecisionTree':
		quantitative = True

	# 
	print 'Training ....'
	bt = time.time()

	dataSet, labels = dataSetBuilder.ReadDataSet(dataFileName)
	purifiedDataSet, minValue, maxValue, ranges = dataSetBuilder.PurifyDataSet(dataSet, quantitative)
	trainedDataSet = classifier.TrainingDataSet(purifiedDataSet)

	at = time.time()
	print 'End Training : %d s' % (at - bt)

	#
	print 'Testing ....'
	bt = time.time()

	testDataSet, testLabels = dataSetBuilder.ReadDataSet(testFileName)
	errorCount = classifier.TestDataSet(trainedDataSet, testDataSet, labels, testLabels, minValue, maxValue, ranges)

	at = time.time()
	print 'End Testing : %d s' % (at - bt)

	print '====================== Result ==========================='
	print 'TotalTest / ErrorTest : %d / %d' % (len(testDataSet), errorCount)
	print 'ErrorPecentage : %.4f' % ( (float(errorCount) / len(testDataSet)) )
	print '=========================================================='


if __name__ == '__main__' :
	if len(sys.argv) != 4:
		print 'USAGE : Main.py [data filename] [test filename] [algorithm name]'
		print 'Example : Main.py adultNumber.data adultNumber.test KNN'
	else:
		DoTrainingAndTest(sys.argv[1], sys.argv[2], sys.argv[3])
