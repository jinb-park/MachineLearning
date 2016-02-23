import operator
import sys
import time
from KNN import KNN
from DecisionTree import DecisionTree
from NaiveBayes import NaiveBayes
from LogisticRegression import LogisticRegression
from SVM import SVM
from AdultDataSet import AdultDataSet
from numpy import *

def GetClassifier(algoName):
	if algoName == 'KNN':
		return KNN()
	elif algoName == 'DecisionTree':
		return DecisionTree()
	elif algoName == 'NaiveBayes':
		return NaiveBayes()
	elif algoName == 'LogisticRegression':
		return LogisticRegression()
	elif algoName == 'SVM':
		return SVM()
	else:
		print 'Not Support Algorithm'
		exit()

def GetDataSetBuilder(dataFileName, classifier):
	if dataFileName.find('adult') > -1:
		return AdultDataSet(classifier)
	else:
		return AdultDataSet(classifier)

def DoTrainingAndTest(dataFileName, testFileName, algoName):
	print 'DataFileName : %s, TestFileName : %s, AlgorithmName : %s' % (dataFileName, testFileName, algoName)
	errorCount = 0

	#
	classifier = GetClassifier(algoName)
	dataSetBuilder = GetDataSetBuilder(dataFileName, classifier)

	# 
	print 'Training ....'
	bt = time.time()

	dataSet, labels, attributes = dataSetBuilder.ReadDataSet(dataFileName)
	purifiedDataSet = dataSetBuilder.PurifyDataSet(dataSet)
	trainedDataSet = classifier.TrainingDataSet(purifiedDataSet, labels, attributes)

	at = time.time()
	print 'End Training : %d s' % (at - bt)

	#
	print 'Testing ....'
	bt = time.time()

	testDataSet, testLabels, attributes = dataSetBuilder.ReadDataSet(testFileName)
	errorCount = classifier.TestDataSet(dataSet, trainedDataSet, testDataSet, labels, testLabels, attributes)

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
