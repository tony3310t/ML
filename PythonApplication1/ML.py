##from sklearn.svm import SVR
import csv
#import jieba
##jieba.suggest_freq('沙瑞金', True)
##jieba.suggest_freq('易学习', True)
##jieba.suggest_freq('王大路', True)
##jieba.suggest_freq('京州', True)

##with open('C:\\Users\\Public\\1.txt') as f:
##    document = f.read()
    
##    #document_decode = document.decode('GBK')
##    document_cut = jieba.cut(document)
##    #print  ' '.join(jieba_cut)
##    result = ' '.join(document_cut)
##    #result = result.encode('utf-8')
##    with open('C:\\Users\\Public\\2.txt', 'w') as f2:
##        f2.write(result)
##f.close()
##f2.close()  

##with open('C:\\Users\\Public\\3.txt', "U", encoding = 'utf-8-sig') as f:
##    document = f.read()
    
##    document_cut = jieba.cut(document)
##    #print  ' '.join(jieba_cut)
##    result = ' '.join(document_cut)
##    with open('C:\\Users\\Public\\5.txt', 'w', encoding = 'utf-8-sig') as f2:
##        f2.write(result)
##f.close()
##f2.close()   

#with open('C:\\Users\\Public\\2.txt') as f3:
#    res1 = f3.read()
#print (res1)

##从文件导入停用词表
#stpwrdpath = "C:\\Users\\Public\\stop_words.txt"
#stpwrd_dic = open(stpwrdpath, 'rb')
#stpwrd_content = stpwrd_dic.read()
##将停用词表转换为list  
#stpwrdlst = stpwrd_content.splitlines()
#stpwrd_dic.close()

#from sklearn.feature_extraction.text import TfidfVectorizer
#corpus = [res1]
#vector = TfidfVectorizer(stop_words=stpwrdlst)
#tfidf = vector.fit_transform(corpus)
#print (tfidf)

#wordlist = vector.get_feature_names()#获取词袋模型中的所有词  
## tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
#weightlist = tfidf.toarray()  
#print(len(wordlist))
##打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
#for i in range(len(weightlist)):  
#    print ("-------第",i,"段文本的词语tf-idf权重------"  )
#    for j in range(len(wordlist)):  
#        print (wordlist[j],weightlist[i][j]  )

import random
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import SQLAcess as sqlAccess
newsList = sqlAccess.GetData("select * from googlenews where stockname='鴻海(2317)'")

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

stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list  
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

instanceList = []
targetList = []
i=0
#while i<len(newsList):
while i<1000:
	featureDict = {}
	for index in range(len(keywordList)):
		featureDict[keywordList[index]] = 0

	content = newsList[i].get('newscontent')

	document_cut = jieba.cut(content)
	#print  ' '.join(jieba_cut)
	result = ' '.join(document_cut)
	corpus = [result]
	vector = TfidfVectorizer(stop_words=stpwrdlst)
	tfidf = vector.fit_transform(corpus)
	#print (tfidf)

	wordlist = vector.get_feature_names()#获取词袋模型中的所有词  
	# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
	weightlist = tfidf.toarray()  
	#print(len(wordlist))
	
	#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
	for j in range(len(weightlist)):  
	    for k in range(len(wordlist)):
	        if wordlist[k] in keywordList:
	            featureDict[wordlist[k]] = float(weightlist[j][k])
	            #print (wordlist[k],weightlist[j][k])
	        else:
	            aa=0
	            #print()
	
	instanceList.append(list(featureDict.values()))
	targetList.append(random.randint(0,1))
	i=i+1

instanceList = np.array(instanceList)
targetList = np.array(targetList)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(instanceList, targetList).predict(instanceList)

for i in range(len(y_rbf)):
	y_rbf[i] = int(round(y_rbf[i]))

X = []
for i in range(len(targetList)): 
	tmp=[]
	tmp.append(i)
	X.append(tmp)
X = np.array(X)

count = 0
for i in range(len(targetList)):
	if targetList[i] == y_rbf[i]:
		count = count +1

plt.scatter(X, targetList, c='k', label='data')
plt.plot(X, y_rbf, c='g', label='RBF model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression, rate = ' + str( round(float(count)/float(len(targetList)), 2) ))
plt.legend()
plt.show()



