import operator
from numpy import *

def IsNumber(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def GetNumOfAttributes(fname):
	fr = open(fname)
	line = fr.readline()
	line = line.strip()
	listFromLine = line.split(' ')
	return len(listFromLine)

def GetNumOfLines(fname):
	fr = open(fname)
	return len(fr.readlines())

def GetMinMaxRanges(dataSet):
	numpyArr = array(dataSet)
	minValue = numpyArr.min(axis=0)
	maxValue = numpyArr.max(axis=0)
	ranges = maxValue - minValue

	return minValue, maxValue, ranges


def normalizeDataSet(dataSet):
	minValue, maxValue, ranges = GetMinMaxRanges(dataSet)

	dataSet = dataSet - tile(minValue, (len(dataSet), 1))
	dataSet = dataSet / tile(ranges, (len(dataSet), 1))

	return dataSet, minValue, maxValue, ranges


class AdultDataSet(object):
	numOfLines = 0
	numOfAttributes = 0

	def ReadDataSet(self, fname):
		self.numOfLines = GetNumOfLines(fname)
		self.numOfAttributes = GetNumOfAttributes(fname)

		fr = open(fname)
		dataSet = []
		classLabelVector = []
		index = 0

		for line in fr.readlines():
			line = line.strip()
			if line.find('?') > -1:
				continue

			listFromLine = line.split(' ')
			lastDataIdx = len(listFromLine) - 1
			dataSet.append(listFromLine[0:lastDataIdx])
			classLabelVector.append(listFromLine[-1])

		for data in dataSet:
			idx = 0
			for attr in data:
				if IsNumber(attr) == True:	# Convert NumberString to Number
					data[idx] = float(attr)	
				idx += 1

		return dataSet, classLabelVector


	def PurifyDataSet(self, dataSet):
		return normalizeDataSet(dataSet)


