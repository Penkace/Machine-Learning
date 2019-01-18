'''
  Author : Peng Kai
  Compiler : Sublime3
  References : Machine Learning in Action
               Jack Cui : https://zhuanlan.zhihu.com/p/28720393
  Problem: 两个问题，第一个是关于正则表达式的问题，第二个问题是输出结果
           第一个问题，在JackCui以及《机器学习实战》中，对于文本内容切分的方法是 '\W*', r'\W*'. 这两种切分方法的意思是选择除了字母、数字和下划线以外
           的任何字符进行划分，并且使用量词 '*' 进行贪婪原则的匹配，即最大程度的选择。但是在实际的情况下却并非按照贪婪的原则，所以我重新编辑了函数。
           第二个问题，是关于结果的，留存交叉检验的结果是100%的错误率，个人并没有发现代码的逻辑错误
  
'''

import numpy as np
import re
import random

def createVocabList(dataSet):
	vocabSet = set([])
	for doc in dataSet:
		vocabSet = set(doc) | vocabSet
	return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print("The world %s is not in my vocabList"%word)
	return returnVec


'''
根据上面生成的词汇表，下面的函数用于构建词袋模型
'''
def bagOfWords2VecMN(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]+=1
	return returnVec

'''
下面是编写朴素贝叶斯训练函数，这里需要注意使用拉普拉斯平滑
'''
def trainNB0(trainMat,trainClass):
	n = len(trainMat)
	m = len(trainMat[0])
	pAbusive = sum(trainClass)/float(n)
	p0Num = np.ones(m);p1Num = np.ones(m)
	p0Denom = 2.0;p1Denom = 2.0
	for i in range(n):
		if trainClass[i]==1:
			p1Num +=trainMat[i]
			p1Denom +=sum(trainMat[i])
		else:
			p0Num +=trainMat[i]
			p0Denom +=sum(trainMat[i])
	p1Vec = np.log(p1Num/p1Denom)
	p0Vec = np.log(p0Num/p0Denom)
	return p0Vec, p1Vec, pAbusive

'''
下面写朴素贝叶斯的分类函数，关于p0与p1，这里遵循的原则是对数相加等于帧数相乘，两对数同底
'''
def classifyNB(vec2,p0V,p1V,pClass):
	p0 = sum(vec2*p0V) + np.log(1.0 - pClass)
	p1 = sum(vec2*p1V) + np.log(pClass)
	if p1 > p0:
		return p1
	else:
		return p0


'''
下面是对文本进行解析，通过使用正则的方法进行文本的切分，这里是问题一
'''
def textParse(bigString):
	listOfTokens = re.split(r'\W',bigString) # 在没有使用量词 '*' 的情况下，可以划分出单词
	
	listOfTokens = [x for x in listOfTokens if x!='']  # 去掉列表中的空格或空字符
	
	list = []
	for word in listOfTokens: # 列表中的长度大于2的单词全部转化为小写并且加入到新的列表list中
		if len(word)>=2:
			word.lower()
		list.append(word)
	return list

'''
测试朴素贝叶斯分类器
'''
def spamTest():
        docList = []
        classList = []
	for i in range(1,26):  # 对所有的样本进行操作，其中spam里面的全部都是垃圾邮件，所以存放的类别是1，其他的是0,顺序是交叉存放的，10101...这样子
		wordList = textParse(open('email/spam/%d.txt'%i,'r').read())
		docList.append(wordList)
		classList.append(1) 
		wordList = textParse(open('email/ham/%d.txt'%i,'r').read())
		docList.append(wordList)
		classList.append(0)

	vocabList = createVocabList(docList)  #创建词汇表

	trainingSet = list(range(50)) # 存放下标的训练集，里面值和下标都是一样的 0-49

	testSet = [] # 用于存放测试集的下标
	for i in range(10): # 从所有的文本中选出10个测试样本
		randIndex = int(random.uniform(0,len(trainingSet))) # uniform是在范围内随机返回一个浮点数，通过random静态调用
		testSet.append(trainingSet[randIndex]) # 测试样本中添加所属的样本
		del(trainingSet[randIndex]) # 从列表中移除测试集对应下标元素，从索引中去掉这10个样本的索引值

	trainMat = [];trainClass = [];
	for docIndex in trainingSet:  # 遍历训练集
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClass.append(classList[docIndex]) # 这个下标对应的类型标签

	p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClass)) # 给出侮辱性文档的概率以及两个类别的概率向量

	errorCount = 0 
	for docIndex in testSet:  # 上面已经求得概率了，就是相应的p(w_i|c_1),下面开始测试错误率
		wordVector = setOfWords2Vec(vocabList,docList[docIndex]) # 先构造词向量  
		if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:  # classList[docIndex]是判断邮件类型的内容
			errorCount+=1 # 如果和实际的情况不符合，那么就算错误
			print("分类错误的测试集:",docList[docIndex])
	print("错误率：%.2f%%"%(float(errorCount)/len(testSet)*100))

if __name__ == '__main__':
	spamTest()


