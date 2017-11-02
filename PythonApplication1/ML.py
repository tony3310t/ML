import numpy as np
import pickle
import csv
import SQLAcess as sqlAccess
import Common as common
import os
import json
import datetime
import matplotlib.pyplot as plt
from  random import randint


#newsList = sqlAccess.GetData("select * from googlenews")
#common.GetFrequency(newsList)

import re
import random
from sklearn.svm import SVR
from sklearn.svm import SVC
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier

jieba.set_dictionary('dict.txt')

stopwordList = []
with open('stopword.txt') as f:
    stopwordList = f.read().splitlines()

stpwrdpath = "dict.txt"
stpwrd_dic = open(stpwrdpath, 'r', encoding='UTF-8')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list  
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

keywordList = []
with open('keywords.csv', newline='', encoding = 'utf-8') as f:		
	reader = csv.reader(f)			
	for row in reader:
		if row == []:
			continue					
		keyword = str(row[0])
				
		try:
			jieba.suggest_freq(keyword, True)
			keywordList.append(keyword)
		except:
			print(keyword)

def segmentWord(content):    
	document_cut = jieba.cut(content)			
	result = ' '.join(document_cut)        
	return result

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

def removeStopword(data):
	final = ''  
	for noun in data:  
		if noun not in stopwordList:  
			final += str(noun)  
	return final  

def trainSVR(data,label):
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_rbf.fit(data,label)
	return svr_rbf

appDataPath = os.getenv('APPDATA')
if os.path.exists(appDataPath + '\StockData') == False:
	os.makedirs(appDataPath + '\StockData')

compList = sqlAccess.GetData("select distinct stockname from googlenews")

#count=0
#for i in range(len(compList)):
#	try:
#		f = open(compList[i].get('stockname') + '_1.txt', 'r', encoding = 'UTF-8')
#		result = f.read()
#		a,b = result.split(':')
#		f.close()

#		f2 = open(compList[i].get('stockname') + '.txt', 'r', encoding = 'UTF-8')
#		result2 = f2.read()
#		c,d = result2.split(':')
#		f2.close()

#		if float(b) - float(d) > 0:
#			print(compList[i].get('stockname'))
#			count = count +1
#	except:
#		print()

#print(float(count) / float(len(len(compList))))

#for compIndex in range(len(compList)):
#	if os.path.exists(compList[compIndex].get('stockname') + "_1.txt") == False:
#		continue
#	if os.path.exists(compList[compIndex].get('stockname')) == False:
#		os.makedirs(compList[compIndex].get('stockname'))

#	f = open(compList[compIndex].get('stockname') + "_1.txt",'r', encoding = 'utf8')
#	data = f.read()
#	f.close()
#	results = data.split(',')

#	for i in range(len(results)):
#		tmp = results[i].split(':')
#		rate = float(tmp[1])*100
#		category = tmp[0].split(')')
#		category = category[1]
#		listA = []
#		listB = []
#		for i in range(100):
#			randNo = randint(0, 1)
#			listA.append(randNo)
#			listB.append(randNo)

#		for i in range(100-int(rate)):
#			randNo = randint(0, 99)

#			if listB[randNo] == 0:
#				listB[randNo] = 1
#			else:
#				listB[randNo] = 0

#		X = []
#		for i in range(100): 
#			tmp=[]
#			tmp.append(i)
#			X.append(tmp)
#		X = np.array(X)
#		fig = plt.figure()
#		plt.scatter(X, listA, c='r', label='Not match')
#		plt.scatter(X, listB, c='g', label='Match')
#		plt.xlabel('Data')
#		plt.ylabel('Label')
#		plt.title(category + ' days of ' + compList[compIndex].get('stockname') + ', rate = ' + str(rate) + '%')
#		plt.legend()
#		#plt.show()
#		fig.savefig(compList[compIndex].get('stockname') + '\\' + category + ',' + str(rate) +'.png')

strMsg = ''
rateDict = {}	
for compIndex in range(len(compList)):
	try:
		#if os.path.exists(compList[compIndex].get('stockname') + "_SVC_Train.txt") == True:
		#	continue
		print(str(compIndex))
		strMsg = ''
		left = compList[compIndex].get('stockname').find('(')+1
		right = compList[compIndex].get('stockname').find(')')
		compNo = compList[compIndex].get('stockname')[left:right]
		#common.GetBasicInfo(compNo)
		data = []
		if os.path.exists(appDataPath + '\StockData\\' + str(compNo)) == True:
			f = open(appDataPath + '\StockData\\' + str(compNo) + '\\BasicInfo.txt','r')
			data = json.loads(f.read())
			f.close()
		if len(data) == 0:
			continue

		newsList = sqlAccess.GetData("select * from googlenews where stockname='"+compList[compIndex].get('stockname')+"' and newsDate < '2017/08/01'")
		segList = []
		#instanceList = []
		target15List = []
		target10List = []
		target5List = []

		if len(newsList) % 2 == 1:
			del newsList[-1]

		if len(newsList) % 10 != 0:
			mod = len(newsList) % 10
			for index in range(mod):
				del newsList[-1]

		instanceCount = len(newsList)
		
		if instanceCount > 1000:
			instanceCount = 1000

		testDataIndexStart = instanceCount*0.8
		testDataIndexStart = int(testDataIndexStart)
		trainDataEndIndex = instanceCount*0.8
		trainDataEndIndex = int(trainDataEndIndex)
		if trainDataEndIndex%10 !=0:
			trainDataEndIndex = trainDataEndIndex-(trainDataEndIndex%10)

		for i in range(instanceCount):
			content = newsList[i].get('newscontent')
			seg = striphtml(content)
			seg = segmentWord(seg)	
			seg = removeStopword(seg)
			segList.append(seg)

			try:
				target15List.append(0)
				target10List.append(0)
				target5List.append(0)

				startDateDict = {}
				endDate15Dict = {}
				endDate10Dict = {}
				endDate5Dict = {}
				Date = ''
				strMsg = '2'
				Date = newsList[i].get('newsDate')
				for index in range(10):
					try:					
						startDateDict = next((item for item in data if item.get("Date") == datetime.datetime.strftime(Date, '%Y/%m/%d')))
						break
					except:
						Date = Date + datetime.timedelta(days=1)

				Date = newsList[i].get('newsDate') + datetime.timedelta(days=15)
				for index in range(10):
					try:					
						endDate15Dict = next((item for item in data if item.get("Date") == datetime.datetime.strftime(Date, '%Y/%m/%d')))
						break
					except:
						Date = Date + datetime.timedelta(days=1)
				strMsg = '3'
				diff = float(endDate15Dict.get('Close')) - float(startDateDict.get('Close'))
			
				if diff >= 0:
					target15List[i] = 1
				else:
					target15List[i] = 0

				Date = newsList[i].get('newsDate') + datetime.timedelta(days=10)
				for index in range(10):
					try:					
						endDate10Dict = next((item for item in data if item.get("Date") == datetime.datetime.strftime(Date, '%Y/%m/%d')))
						break
					except:
						Date = Date + datetime.timedelta(days=1)
				strMsg = '3'
				diff = float(endDate10Dict.get('Close')) - float(startDateDict.get('Close'))
			
				if diff >= 0:
					target10List[i] = 1
				else:
					target10List[i] = 0

				Date = newsList[i].get('newsDate') + datetime.timedelta(days=5)
				for index in range(10):
					try:					
						endDate5Dict = next((item for item in data if item.get("Date") == datetime.datetime.strftime(Date, '%Y/%m/%d')))
						break
					except:
						Date = Date + datetime.timedelta(days=1)
				strMsg = '3'
				diff = float(endDate5Dict.get('Close')) - float(startDateDict.get('Close'))
			
				if diff >= 0:
					target5List[i] = 1
				else:
					target5List[i] = 0
				strMsg = '4'
				#instanceList.append(list(featureDict.values()))
			except:
				a=0
						
		vectorizer=CountVectorizer()
		tfidftransformer=TfidfTransformer()
		tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(segList))  # 先转换成词频矩阵，再计算TFIDF值
		weight = tfidf.toarray()
		print(tfidf.shape)
		print("len of weight:" + str(len(weight)) + "," + "len of target15:" + str(len(target15List))+ "," + "len of target10:" + str(len(target10List))+ "," + "len of target5:" + str(len(target5List)))
		target15List = np.array(target15List)
		target10List = np.array(target10List)
		target5List = np.array(target5List)
		
		splitWeightList = []
		splitTarget15List = []
		splitTarget10List = []
		splitTarget5List = []
		for i in range(10):
			period = trainDataEndIndex / 10
			period = int(period)
			splitData = weight[i*period:(i+1)*period]
			splitWeightList.append(splitData)
			splitTarget = target15List[i*period:(i+1)*period]
			splitTarget15List.append(splitTarget)
			splitTarget = target10List[i*period:(i+1)*period]
			splitTarget10List.append(splitTarget)
			splitTarget = target5List[i*period:(i+1)*period]
			splitTarget5List.append(splitTarget)

		finalModel15 = trainSVR(splitWeightList[0], splitTarget15List[0])
		finalModel10 = trainSVR(splitWeightList[0], splitTarget10List[0])
		finalModel5 = trainSVR(splitWeightList[0], splitTarget5List[0])
		tmpRate15 = -1
		tmpRate10 = -1
		tmpRate5 = -1
		for i in range(10):			
			validData = splitWeightList[i]
			validLabel15 = splitTarget15List[i]
			validLabel10 = splitTarget10List[i]
			validLabel5 = splitTarget5List[i]
			trainData = splitWeightList[0]
			trainLabel15 = splitTarget15List[0]
			trainLabel10 = splitTarget10List[0]
			trainLabel5 = splitTarget5List[0]
			firstIndex = 0

			if i == 0:
				trainData = splitWeightList[1]
				trainLabel15 = splitTarget15List[1]
				trainLabel10 = splitTarget10List[1]
				trainLabel5 = splitTarget5List[1]
				firstIndex = 1

			for j in range(10):
				if j == i or j == firstIndex:
					continue
				trainData = np.append(trainData,splitWeightList[j], axis=0)
				trainLabel15 = np.append(trainLabel15, splitTarget15List[j], axis=0)
				trainLabel10 = np.append(trainLabel10, splitTarget10List[j], axis=0)
				trainLabel5 = np.append(trainLabel5, splitTarget5List[j], axis=0)
			

			model15 = trainSVR(trainData, trainLabel15)
			result = model15.predict(validData)

			for index in range(len(result)):
				result[index] = int(round(result[index]))

			count = 0
			for index in range(len(validLabel15)):
				if result[index] == validLabel15[index]:
					count = count + 1

			rate = float(count)/float(len(validLabel15))
			if rate > tmpRate15:
				finalModel15 = model15
				tmpRate15 = rate

			model10 = trainSVR(trainData, trainLabel10)
			result = model10.predict(validData)

			for index in range(len(result)):
				result[index] = int(round(result[index]))

			count = 0
			for index in range(len(validLabel10)):
				if result[index] == validLabel10[index]:
					count = count + 1

			rate = float(count)/float(len(validLabel10))
			if rate > tmpRate10:
				finalModel10 = model10
				tmpRate10 = rate

			model5 = trainSVR(trainData, trainLabel5)
			result = model5.predict(validData)

			for index in range(len(result)):
				result[index] = int(round(result[index]))

			count = 0
			for index in range(len(validLabel5)):
				if result[index] == validLabel5[index]:
					count = count + 1

			rate = float(count)/float(len(validLabel5))
			if rate > tmpRate5:
				finalModel5 = model5
				tmpRate5 = rate

		with open(compList[compIndex].get('stockname') + '\\' + str(tmpRate15) + '_15_SVR.pickle', 'wb') as f:
			pickle.dump(finalModel15, f)

		with open(compList[compIndex].get('stockname') + '\\' + str(tmpRate10) + '_10_SVR.pickle', 'wb') as f:
			pickle.dump(finalModel10, f)

		with open(compList[compIndex].get('stockname') + '\\' + str(tmpRate5) + '_5_SVR.pickle', 'wb') as f:
			pickle.dump(finalModel5, f)

		result = finalModel15.predict(weight[testDataIndexStart:])
		for index in range(len(result)):
			result[index] = int(round(result[index]))

		count = 0
		for index in range(len(result)):
			if result[index] == target15List[index+testDataIndexStart]:
				count = count + 1

		rate = float(count)/float(len(result))

		rate = rate*100
		category = '15'
		listA = []
		listB = []
		for i in range(100):
			randNo = randint(0, 1)
			listA.append(randNo)
			listB.append(randNo)

		for i in range(100-int(rate)):
			randNo = randint(0, 99)

			if listB[randNo] == 0:
				listB[randNo] = 1
			else:
				listB[randNo] = 0

		X = []
		for i in range(100): 
			tmp=[]
			tmp.append(i)
			X.append(tmp)
		X = np.array(X)
		fig = plt.figure()
		plt.scatter(X, listA, c='r', label='Not match')
		plt.scatter(X, listB, c='g', label='Match')
		plt.xlabel('Data')
		plt.ylabel('Label')
		plt.title(category + ' days of ' + compList[compIndex].get('stockname') + ', rate = ' + str(rate) + '%')
		plt.legend()
		#plt.show()
		fig.savefig(compList[compIndex].get('stockname') + '\\' + str(rate) + '_' + category + '_SVR_test' +'.png')

		
		#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
		#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
  #                  hidden_layer_sizes=(5, 2), random_state=1)
		#y_rbf15 = clf.fit(weight[:int(len(weight)*0.8)], target15List[:int(len(weight)*0.8)]).predict(weight[int(len(weight)*0.8):])
		#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
  #                  hidden_layer_sizes=(5, 2), random_state=1)
		#y_rbf10 = clf.fit(weight[:int(len(weight)*0.8)], target10List[:int(len(weight)*0.8)]).predict(weight[int(len(weight)*0.8):])
		#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
  #                  hidden_layer_sizes=(5, 2), random_state=1)
		#y_rbf5 = clf.fit(weight[:int(len(weight)*0.8)], target5List[:int(len(weight)*0.8)]).predict(weight[int(len(weight)*0.8):])
		#strMsg = '5'
		#for i in range(len(y_rbf15)):
		#	y_rbf15[i] = int(round(y_rbf15[i]))

		#for i in range(len(y_rbf10)):
		#	y_rbf10[i] = int(round(y_rbf10[i]))

		#for i in range(len(y_rbf5)):
		#	y_rbf5[i] = int(round(y_rbf5[i]))

		#strMsg = '6'
		#rate15 = ''
		#count = 0
		#halfIndex = int(len(target15List)*0.8)
		#BackIndex = int(len(target15List)*0.2)
		#for i in range(BackIndex):
		#	if target15List[i+halfIndex] == y_rbf15[i]:
		#		count = count +1

		#rate15 = str(round(float(count)/float((len(target15List)*0.2)), 2))		
		#print(compList[compIndex].get('stockname') + " of 15:" + rate15)

		#rate10 = ''
		#count = 0
		#halfIndex = int(len(target10List)*0.8)
		#for i in range(BackIndex):
		#	if target10List[i+halfIndex] == y_rbf10[i]:
		#		count = count +1

		#rate10 = str(round(float(count)/float((len(target10List)*0.2)), 2))		
		#print(compList[compIndex].get('stockname') + " of 10:" + rate10)

		#rate5 = ''
		#count = 0
		#halfIndex = int(len(target5List)*0.8)
		#for i in range(BackIndex):
		#	if target5List[i+halfIndex] == y_rbf5[i]:
		#		count = count +1

		#rate5 = str(round(float(count)/float((len(target5List)*0.2)), 2))		
		#print(compList[compIndex].get('stockname') + " of 5:" + rate5)
		
		#file = open(compList[compIndex].get('stockname')+'_MLP.txt', 'w', encoding = 'UTF-8')    # 也可使用指定路徑等方式，如： C:\A.txt
		#file.write(str(compList[compIndex].get('stockname') + "15:" + rate15 + "," + compList[compIndex].get('stockname') + "10:" + rate10 + "," + compList[compIndex].get('stockname') + "5:" + rate5))	
		#file.close()

		#newsList = sqlAccess.GetData("select * from googlenews where stockname='"+compList[compIndex].get('stockname')+"' and newsDate < '2017/08/01'")
		#instanceList = []
		#targetList = []
		#i=0
		#instCount = len(newsList)
		#if instCount > 100:
		#	instCount = 100

		#if instCount % 2 !=0:
		#	instCount = instCount -1

		#strMsg = '1'
		#while i<instCount:
		#	featureDict = {}
		#	for index in range(len(keywordList)):
		#		featureDict[keywordList[index]] = 0

		#	content = newsList[i].get('newscontent')

		#	document_cut = jieba.cut(content)
		#	#print  ' '.join(jieba_cut)
		#	result = ' '.join(document_cut)
		#	corpus = [result]
		#	vector = TfidfVectorizer(stop_words=stpwrdlst)
		#	tfidf = vector.fit_transform(corpus)
		#	#print (tfidf)

		#	wordlist = vector.get_feature_names()#获取词袋模型中的所有词  
		#	# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
		#	weightlist = tfidf.toarray()  
		#	#print(len(wordlist))
	
		#	#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
		#	for j in range(len(weightlist)):  
		#		for k in range(len(wordlist)):
		#			if wordlist[k] in keywordList:
		#				featureDict[wordlist[k]] = float(weightlist[j][k])
		#				#print (wordlist[k],weightlist[j][k])
					
		#	try:
		#		startDateDict = {}
		#		endDateDict = {}
		#		Date = ''
		#		strMsg = '2'
		#		Date = newsList[i].get('newsDate')
		#		for index in range(10):
		#			try:					
		#				startDateDict = next((item for item in data if item.get("Date") == datetime.datetime.strftime(Date, '%Y/%m/%d')))
		#				break
		#			except:
		#				Date = Date + datetime.timedelta(days=1)

		#		Date = newsList[i].get('newsDate') + datetime.timedelta(days=15)
		#		for index in range(10):
		#			try:					
		#				endDateDict = next((item for item in data if item.get("Date") == datetime.datetime.strftime(Date, '%Y/%m/%d')))
		#				break
		#			except:
		#				Date = Date + datetime.timedelta(days=1)
		#		strMsg = '3'
		#		diff = float(endDateDict.get('Close')) - float(startDateDict.get('Close'))
			
		#		if diff >= 0:
		#			targetList.append(1)
		#		else:
		#			targetList.append(0)
		#		strMsg = '4'
		#		instanceList.append(list(featureDict.values()))
		#	except:
		#		a=0

		#	i=i+1

		
		#instanceList = np.array(instanceList)
		#targetList = np.array(targetList)
		#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
		#y_rbf = svr_rbf.fit(instanceList[:int(len(instanceList)/2)], targetList[:int(len(instanceList)/2)]).predict(instanceList[int(len(instanceList)/2):])
		#strMsg = '5'
		#for i in range(len(y_rbf)):
		#	y_rbf[i] = int(round(y_rbf[i]))

		#X = []
		#for i in range(len(targetList)): 
		#	tmp=[]
		#	tmp.append(i)
		#	X.append(tmp)
		#X = np.array(X)
		#strMsg = '6'
		#count = 0
		#halfIndex = int(len(targetList)/2)
		#for i in range(halfIndex):
		#	if targetList[i+halfIndex] == y_rbf[i]:
		#		count = count +1
		
		#print(compList[compIndex].get('stockname') + ":" + str(round(float(count)/float((len(targetList)/2)), 2)))
		#strMsg = '7'

		#file = open(compList[compIndex].get('stockname')+'_1.txt', 'w', encoding = 'UTF-8')    # 也可使用指定路徑等方式，如： C:\A.txt
		#file.write(str(compList[compIndex].get('stockname') + ":" + str(round(float(count)/float((len(targetList)/2)), 2))))	
		#file.close()

		
		##print(str(round(float(count)/float((len(targetList)/2)), 2)))
		##plt.scatter(X, targetList, c='k', label='data')
		##plt.plot(X, y_rbf, c='g', label='RBF model')
		##plt.xlabel('data')
		##plt.ylabel('target')
		##plt.title('SVR of '+compList[compIndex].get('stockname')+', rate = ' + str( round(float(count)/float(len(targetList)), 2) ))
		##plt.legend()
		##plt.show()
	except Exception as e:
		print(compList[compIndex].get('stockname') + ":" + 'Failed to upload to ftp: '+ str(e) + ":" + strMsg)






