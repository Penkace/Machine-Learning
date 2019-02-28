# 这是第一部分的代码，关于支持函数的测试
from numpy import *

def loadDataSet(filename):
	dataMat = []
	fr = open(filename)
	for line in fr.readlines():
		curLine = line.split()
		# 将float作用于curLine的每一个元素
		# fltLine = map(float,curLine)
		fltLine = [float(word) for word in curLine]
		dataMat.append(fltLine)
	fr.close()
	return dataMat

def calcudist(vecA,vecB):
	return sqrt(sum(power(vecA-vecB,2))) # power(a,b)函数是指对a中的每个元素求b次方

def randCent(dataSet,k):
	n = shape(dataSet)[1] # 返回列数
	centerpoint = mat(zeros((k,n))) # 创建质点矩阵
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j]) - minJ)
		centerpoint[:,j] = mat(minJ + rangeJ*random.rand(k,1))
	return centerpoint



if __name__ =='__main__':
	dataMatrix = mat(loadDataSet('testSet.txt'))
	print(calcudist(dataMatrix[0],dataMatrix[1]))
	centerP = randCent(dataMatrix,2)
	print(centerP)

