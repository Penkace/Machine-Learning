# 通常我们使用误差平方和，即SSE(Sum of Squared Error)来度量聚类效果的指标，SSE值越小表示数据点越接近于它们的质心，聚类效果好
# 如何改进聚类的效果，一种方法是将具有最大SSE值的簇分成两个，将最大簇包含的点过滤出来并在这些点上运行K-均值算法。这样的代价是需要把两个簇进行合并。
# 上述的方法在低维的时候可以使用，对于高维的情况，计算所有质点的之间的距离，然后合并距离最近的两个点；第二种方法是合并两个簇，计算总SSE值，在所有的两个簇上重复知道找到合并最佳的两个簇位置。
# 下面我们将讨论二分K-均值算法，一种克服均值算法收敛到局部最优解的方法。思路是将有点视为一个簇，然后将簇一分为二。之后选择一个簇继续进行划分，选择哪一个簇进行划分取决于划分后是否可以最大程度降低SSE值。
# 上述基于SSE划分的过程不断重复直到划分k个簇。

import numpy as np

def loadDataSet(filename):
	dataMat = []
	fr = open(filename)
	for line in fr.readlines():
		curLine = line.split()
		fltLine = [float(word) for word in curLine]
		dataMat.append(fltLine)
	fr.close()
	return dataMat

def calcudist(vecA,vecB):
	return np.sqrt(np.sum(np.power(vecA - vecB,2)))

def randCent(dataSet,k):
	n = np.shape(dataSet)[1]
	centerpoint = np.mat(np.zeros((k,n)))
	for j in range(n): # 求每一列的结果
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j])-minJ)
		centerpoint[:,j] = minJ + rangeJ * np.random.rand(k,1)
	return centerpoint

def KMeans(dataSet,k,distMeas = calcudist,createCent = randCent):
	m = np.shape(dataSet)[0]
	clusterAssment  = np.mat(np.zeros((m,2)))
	centerpoint = createCent(dataSet,k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = float('inf')
			minIndex = -1
			for j in range(k):
				distIJ = distMeas(centerpoint[j,:],dataSet[i,:])
				if distIJ<minDist:
					minDist = distIJ
					minIndex = j
			if clusterAssment[i,0] != minIndex:
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist**2
		print(centerpoint)
		for cent in range(k):
			Sameclu = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
			centpoint[cent,:] = np.mean(Sameclu,axis=0)
	return centerpoint,clusterAssment

def biKmeans(dataSet,k,distMeas = calcudist):
	m = np.shape(dataSet)[0] # 返回数据的总数
	clusterAssment = np.mat(np.zeros((m,2)))  # 构建一个综述的矩阵
	# centerpoint = np.mean(dataSet,axis = 0).tolist()[0]  # 将数组数据的副本作为list返回，这里取list的第一个数据，就是一组数，因为tolist之后返回的是[[-0.111，-0.111]]这种类型的数据
	# cenList = [centerpoint] # 这句和上面的一句可以直接合并
	centList = np.mean(dataSet,axis = 0).tolist() # 得到所有质心的均值，即第一个簇
	# centList用来保留所有的质心
	centerpoint = centList[0]
	# 这里取出第一个质心，刚开始只有初始簇
	for j in range(m): # 计算所有数据点和第一个质心的距离误差
		clusterAssment[j,1] = distMeas(np.mat(centerpoint),dataSet[j,:])**2
	# 开始循环划分簇
	while(len(centList)<k): #小于需要划分的簇的数量
		lowestSSE = float('inf') # 误差平方和设置为无穷大
		for i in range(len(centList)): #遍历已经划分的簇，即所有的质心
			ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i),:] # 找出第一个簇的质心，即该簇的所有数据点。这里返回80×2
			centerMat,splitClustAss = KMeans(ptsInCurrCluster,2,distMeas) # 把第一个簇划分成两个簇
			sseSplit = np.sum(splitClustAss[:,1]) # 返回两个簇的总误差
			sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A != i)[0],1])# .A将矩阵转换成numpy的array数组，剩下的数据点的总误差
			print("sseSplit, and notSplit:",sseSplit,sseNotSplit)
			if (sseSplit + sseNotSplit) < lowestSSE: # 比较划分之后的总误差和最小误差，如果这次划分得到的误差结果减小，则表明划分
				bestCentToSplit = i # 这次划分的簇类型
				bestNewCents = centerMat # 新的质心
				bestClustAss = splitClustAss.copy() # 分配结果的副本
				lowestSSE =  sseSplit + sseNotSplit
		#前面的for循环决定了这次的2分划分保存，下面就是更新，通过过滤器更新
		# 首先从划分结果里面找到0,1的数据点，将其所述的分类，首先将簇分配结果中等于1的划分为centList的长度，等于0的则为
		bestClustAss[np.nonzero(bestClustAss[:,0].A==1)[0],0]= len(centList)
		# 等于0 的则划分到当前的簇分类中，即i
		bestClustAss[np.nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit
		print("the bestCentToSplit is : ",bestCentToSplit)
		print("the len of bestClustAss is : ",len(bestClustAss))
		cenList[bestCentToSplit] = bestNewCents[0,:] # 当前质心更新为新划分的两个簇的一个质心
		cenList.append(bestNewCents[1,:]) # 另外一个划分的质心则新加入到centList中
		clusterAssment[np.nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:] = bestClustAss # 把新划分的簇的下标更新
	return np.mat(cenList),clusterAssment

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

if __name__ == '__main__':
	dataMat = np.mat(loadDataSet("testSet.txt"))
	myCentpoint,clusterAssing = biKmeans(dataMat,4)
	print("质心为：\n",myCentpoint)
	print("聚类结果为：\n",clusterAssing)
	picResult(myCentpoint,clusterAssing,dataMat)
	