import operator
from numpy import *

def stumpClassify(dataMat, dimen, threshVal, threshIneq):
	retArr = ones( (shape(dataMat)[0], 1) )
	if threshIneq == 'lt':
		retArr[dataMat[:, dimen] <= threshVal] = -1.0
	else:
		retArr[dataMat[:, dimen] > threshVal] = -1.0
	return retArr

# generate classifier
def buildStump(dataArr, classLabels, D):
	dataMat = mat(dataArr); labelMat = mat(classLabels).T
	m, n = shape(dataMat)

	bestStump = {}; bestClasEst = mat(zeros((m,1)))
	minError = inf
	for i in range(n):
		stepSize = 0.1
		for j in range(-1, 11):
			for inequal in ['lt', 'gt']:
				threshVal = (0.0 + float(j) * stepSize)
				predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
				errArr = mat(ones((m,1)))
				errArr[predictedVals == labelMat] = 0
				weightedError = D.T * errArr

				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
	weakClassArr = []
	m = shape(dataArr)[0]
	D = mat(ones((m,1)) / m)
	aggClassEst = mat(zeros((m,1)))

	for i in range(numIt):
		bestStump, error, classEst = buildStump(dataArr, classLabels, D)
		#print 'D: ', D.T
		alpha = float(0.5 * log((1.0-error) / max(error, 1e-16 )))
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		#print 'classEst: ', classEst.T

		expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
		D = multiply(D, exp(expon))
		D = D / D.sum()

		aggClassEst += alpha * classEst  # trust
		#print 'aggClassEst: ', aggClassEst.T
		aggErros = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
		errorRate = aggErros.sum() / m
		#print 'total error: ', errorRate, '\n'
		if errorRate == 0.0:
			break
	return weakClassArr

def adaClassify(datToClass, classifierArr):
	datMat = mat(datToClass)
	m = shape(datMat)[0]
	aggClassEst = mat(zeros((m, 1)))

	for i in range(len(classifierArr)):
		classEst = stumpClassify(datMat, classifierArr[i]['dim'], \
					classifierArr[i]['thresh'], classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha'] * classEst
		#print aggClassEst
	return sign(aggClassEst)

class Adaboost(object):

	# return classfierArray
	def TrainingDataSet(self, purifiedDataSet, labels, attributes):

		# convert labels --> +1 and -1
		idx = 0
		for data in labels:
			if float(data) == float(0):
				labels[idx] = float(-1.0)
			else:
				labels[idx] = float(1.0)
			idx += 1

		classifierArr = adaBoostTrainDS(purifiedDataSet, labels, 60)
		return classifierArr

	# trainedDataSet - classifierArray
	def TestDataSet(self, origDataSet, trainedDataSet, testDataSet, trainedLabels, testLabels, attributes):

		testDataCount = shape(testDataSet)[0]
		classifierArr = trainedDataSet

		print 'classifier count : ', shape(classifierArr)[0]

		# convert labels --> +1 and -1
		idx = 0
		for data in testLabels:
			if float(data) == float(0):
				testLabels[idx] = float(-1.0)
			else:
				testLabels[idx] = float(1.0)
			idx += 1

		predictArr = adaClassify(testDataSet, classifierArr)
		errArr = mat(ones((testDataCount, 1)))
		errorCount = errArr[predictArr != mat(testLabels).T].sum()

		return errorCount