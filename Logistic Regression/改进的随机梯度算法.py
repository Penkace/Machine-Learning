'''
------------------------------------------------------尊重原创，抵制抄袭----------------------------------------------------------------
References: 《机器学习实战》第五章  以及  Jack Cui的知乎专栏：https://zhuanlan.zhihu.com/p/29073560    
Problems: 在gradAscent1函数，即改进的随机梯度上升算法，这里根据自己的理解做了一些改动，主要是针对del(dataIndex[randIndex])的删除以及randIndex
          在函数中的引用，每次删除的是dataIndex中对应的randIndex值，在dataIndex中，样本的下标不断的删除，但是在实际的引用是引用randIndex，这样的
          结果是每次生成的随机数的范围在不断的缩小并且重复数字的频率增高，dataMatrix[randIndex]每次引用的样本也是一样的。
          改进的随机梯度上升算法的初衷是尽量减少来回的波动以及加快收敛速度，采用随机选取样本的机制，减少周期性的波动。我个人的理解是随机遍历所有
	  样本，所以才改成dataMat[dataIndex[randIndex]]。但是书上以及博客都是使用dataMat[randIndex],希望朋友们能够指点迷津。
'''

import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt

'''
函数说明： 加载数据，源数据的1,2列加入到dataMat列表中，标签加入到classMat之中
'''
def loadDataSet():
	dataMat = []
	classMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineAr = line.split() # split默认以空格为切割点，包含\n换行符
		dataMat.append([1.0,float(lineAr[0]),float(lineAr[1])])  # dataMat的1,2分别是x,y
		classMat.append(int(lineAr[2])) # 在列表里面类型是字符型而不是数值型
	fr.close()
	return dataMat,classMat


'''
这是sigmoid函数，在numpy中的exp函数能够接受矩阵的参数输入
'''
def sigmoid(inX): # numpy的exp函数支持矩阵运算
	return 1.0/(1+np.exp(-inX))


'''
梯度上升函数，这是最开始的，但是需要计算dataMatrix*theta,计算量庞大，对于数据量的要求比较高，O（maxCycles*m*n）时间复杂度？大概在百量到千量数据范围
'''
def gradAscent(dataMatIn,classLabels):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose() #所有的数组、列表都转换成矩阵
	# print(labelMat.shape) 100x1
	m,n = np.shape(dataMatrix) # 得到dataMatrix的维度
	# print(m,n) 100x3
	alpha = 0.001
	maxCycles = 500
	theta = np.ones((n,1)) 
	# print(sigmoid(dataMatrix).shape) 100x3
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*theta) # 100x3 3x1  = 100x1 dataMtrix * theta计算的次数太多了
		error = labelMat - h # 100x1 - 100x1
		theta = theta + alpha*dataMatrix.transpose()*error # 3x1 + 1x1 x 3x100 x 100x1 = 3x1
	return theta.getA() # 把矩阵转换成数组，返回权重数组



'''
随机梯度上升算法，改进的梯度上升算法，我在这里的改动是引用改成dataIndex[randIndex]，这样可以随机的选取样本，而不是引用随机产生数,注释是方便测试
'''
def gradAscent1(dataMatrix,classMat,Iter = 150):
	m , n = np.shape(dataMatrix)
	theta = np.ones(n)
	for j in range(Iter):
		dataIndex = list(range(m)) # 选取下标
		for i in range(m):
			alpha = 4/(1.0 +j +i)+0.01
			randIndex = int(random.uniform(0,len(dataIndex))) # 随机选取一个样本
			print(randIndex)
			h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*theta)) # 结果是1xn，求出每一列的总和，再代进去求总和
			error = classMat[dataIndex[randIndex]] - h # 1xn - 1xn 的得到error误差 
			theta = theta + alpha * error * dataMatrix[dataIndex[randIndex]]# 1xn + 1x1 x 1xn x 1xn
			# print(theta)
			del(dataIndex[randIndex])
			# print(dataIndex)
	return theta

def plotBestFit(theta):
	dataMat,classMat = loadDataSet()
	dataAr = np.array(dataMat)
	n = np.shape(dataMat)[0]
	xcord1 = []
	ycord1 = []
	xcord2 = []
	ycord2 = []
	for i in range(n):
		if int(classMat[i]) == 1:
			xcord1.append(dataAr[i,1])
			ycord1.append(dataAr[i,2])
		else:
			xcord2.append(dataAr[i,1])
			ycord2.append(dataAr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=20,c='red',marker = 's',alpha = .5)
	ax.scatter(xcord2,ycord2,s=20,c='green',alpha = .5)
	x = np.arange(-3.0,0.3,0.1)
	y = (-theta[0]-theta[1]*x)/theta[2]
	ax.plot(x,y)
	plt.title('BestFit')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()

if __name__ =='__main__':
	dataMat,classMat = loadDataSet()
	theta = gradAscent1(np.array(dataMat),classMat)
	plotBestFit(theta)
  print(theta)
