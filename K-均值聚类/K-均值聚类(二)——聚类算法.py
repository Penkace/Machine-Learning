#下面是K均值算法的代码
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
	dataMat = []
	fr = open(filename)
	for line in fr.readlines():
		curLine = line.split()
		fltLine = [float(word) for word in curLine]
		dataMat.append(fltLine)
	return dataMat

# numpy的matrix类型是可以进行矩阵或者是向量的加减
def calcudist(vecA,vecB):
	return np.sqrt(np.sum(np.power(vecA - vecB,2))) # numpy中的sum函数比较特殊，如果只有相加的参数，那么就是全部相加，加axis=0表示列相加，axis=1表示行相加

'''
函数说明： 该函数为数据集构建一个包含k个随机质心的集合。
这个质心肯定在整个数据集的边界之内。
'''
def randCent(dataSet,k):
	n = np.shape(dataSet)[1] 
	centerpoint = np.mat(np.zeros((k,n)))
	for j in range(n):
		minJ = min(dataSet[:,j]) # 找出第j列最小的元素
		rangeJ = float(max(dataSet[:,j]) - minJ)
		centerpoint[:,j] = minJ + rangeJ * np.random.rand(k,1) # 生成[0,1]的两行一列矩阵,这里先填充centerpoint的第i行
	return centerpoint



'''
函数说明：
clusterAssment : 存储数据集中各点的簇分配结果，第一列是记录簇索引值，第二列是存储当前点到簇质心的距离，这个值后面用来评估聚类效果
这个函数的终止条件是所有数据点的簇分配结果不再改变为止，即clusterChanged为False

'''
def KMeans(dataSet,k,distMeas = calcudist, creatCent = randCent):
	m = np.shape(dataSet)[0] # 返回行数
	clusterAssment = np.mat(np.zeros((m,2)))
	centerpoint = creatCent(dataSet,k)
	# print(centerpoint)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = float('inf')
			minIndex = -1
			for j in range(k):
				distJI = distMeas(centerpoint[j,:],dataSet[i,:]) #计算所有的数据点和质心之间的距离，找出最小点
				if distJI < minDist:
					minDist = distJI
					minIndex = j
			if clusterAssment[i,0]!=minIndex: # 改变第i行即第i个数据点的簇分配索引，知道遍历了所有的数据点这个clusterAssment都不改变了，这个clusterChanged就为False
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist**2
		print(centerpoint)
		for cent in range(k):
			# 下面一句是挑选出簇分配相同的数据点的x的坐标值
			ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]] # numpy 中的nonzero返回非零元素的目录。返回值为元组， 两个值分别为两个维度， 包含了相应维度上非零元素的目录值。   可以通过a[nonzero(a)]来获得所有非零值
			# print(ptsInClust)
			centerpoint[cent,:] = np.mean(ptsInClust,axis = 0) # 前面得到所有簇分配结果相同的数据点，通过mean求平均值
			# print(np.mean(ptsInClust,axis = 0))
			# print(centerpoint[cent:])
	return centerpoint, clusterAssment

# 画图部分
def picResult(myCentpoint,clusterAssing,dataSet):
	X = np.array(myCentpoint[:,0]) # 选择质心的x坐标,matrix有约束，所以需要转化成numpy的数组才好画散点图
	Y = np.array(myCentpoint[:,1]) # 选择质心的y坐标
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_title('Scatter Plot')
	plt.xlabel('X')
	plt.ylabel('Y')
	ax1.scatter(X,Y,c='r',marker= '+',s=300)

	scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
	for i in range(4):
		ptsInCurrCluster = dataSet[np.nonzero(clusterAssing[:,0].A==i)[0],:]
		markerStyle = scatterMarkers[i%len(scatterMarkers)]
		ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],ptsInCurrCluster[:,1].flatten().A[0],marker = markerStyle,s=50)
	plt.show()


if __name__ =='__main__':
	dataMat = np.mat(loadDataSet('testSet.txt'))
	print(dataMat)
	myCentpoint,clusterAssing = KMeans(dataMat,4)
	print("质心为：\n",myCentpoint)
	print("聚类结果：\n",clusterAssing)
	picResult(myCentpoint,clusterAssing,dataMat)

