import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math
import time
import datetime
import h5py
import os
from MongoDB import *
from sklearn import preprocessing
from keras.models import load_model
import csv

dataframeFolderName = 'dataframe'
diagramFolderName = 'lstm_train_diagram_100'
diagramPictureName = '_LSTM_prediction_diagram'
modelFolderName = 'selected_model'
modelFileName = '_LSTM_prediction'
correctRateType = '5_features_lstm_predict_100'
outputFolderName = 'output'
outputEveryUpDownFolderName = 'outputEveryUpDown'

def get_stock_data(stock_name, normalize=True):
	start = datetime.datetime(1990, 1, 1)
	end = datetime.date.today()
	#df = web.DataReader(stock_name, "yahoo", start, end)
	cwd = os.getcwd()
	file_name = cwd + '\\' + dataframeFolderName + '\\' + stock_name
	df = pd.read_pickle(file_name)
	df.drop(['Close'], 1, inplace=True)

	if normalize:        
		min_max_scaler = preprocessing.MinMaxScaler()
		df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))
		df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))
		df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))
		df['Volume'] = min_max_scaler.fit_transform(df.Volume.values.reshape(-1,1))
		df['Adj Close'] = min_max_scaler.fit_transform(df['Adj Close'].values.reshape(-1,1))
	return df

def get_real_stock_data(stock_name, normalize=True):
	start = datetime.datetime(1990, 1, 1)
	end = datetime.date.today()
	#df = web.DataReader(stock_name, "yahoo", start, end)
	cwd = os.getcwd()
	file_name = cwd + '\\' + dataframeFolderName + '\\' + stock_name
	df = pd.read_pickle(file_name)
	
	return df

def load_data(stock, seq_len):
	amount_of_features = len(stock.columns)
	data = stock.as_matrix() 
	sequence_length = seq_len + 1 # index starting from 0
	result = []

	for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
		result.append(data[index: index + sequence_length]) # index : index + 22days

	result = np.array(result)
	row = round(0.9 * result.shape[0]) # 90% split

	train = result[:4273, :] # 90% date
	X_train = train[:, :-1] # all data until day m
	y_train = train[:, -1][:,-1] # day m + 1 adjusted close price

	X_test = result[4273:, :-1]
	y_test = result[4273:, -1][:,-1] 

	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

	return [X_train, y_train, X_test, y_test]

def model_score(model, X_train, y_train, X_test, y_test):
#其實這裡只需要傳入X_test和y_test即可，只是為了和上面格式保持一致因而也將X_train和y_train傳入了
	y_hat = model.predict(X_test)
	y_t = y_test.reshape(-1,1)

	#temp = pd.DataFrame(y_hat)  
	#temp['yhat'] = y_hat
	#temp['y'] = y_t
	#temp_rmse = sqrt(mean_squared_error(temp.y,temp.yhat))
	#temp_mse = mean_squared_error(temp.y,temp.yhat)
	#print('TEMP RMSE: %.3f' % temp_rmse)
	#print('TEMP MSE: %.3f' % temp_mse)
	return y_hat

def denormalize(stock_name, normalized_value):
	start = datetime.datetime(1990, 1, 1)
	end = datetime.date.today()
	#df = web.DataReader(stock_name, "yahoo", start, end)
	cwd = os.getcwd()
	file_name = cwd + '\\' + dataframeFolderName + '\\' + stock_name
	df = pd.read_pickle(file_name)

	df = df['Adj Close'].values.reshape(-1,1)
	normalized_value = normalized_value.reshape(-1,1)

	#return df.shape, p.shape
	min_max_scaler = preprocessing.MinMaxScaler()
	a = min_max_scaler.fit_transform(df)
	new = min_max_scaler.inverse_transform(normalized_value)
	return new

def plot_result(stock_name, normalized_value_p, normalized_value_y_test):
	newp = denormalize(stock_name, normalized_value_p)
	newy_test = denormalize(stock_name, normalized_value_y_test)
	#fig = plt.figure()
	#plt.plot(newp, color='red', label='Prediction')
	#plt.plot(newy_test,color='blue', label='Actual')
	#plt.legend(loc='best')
	#plt.title('The test result for {}'.format(stock_name))
	#plt.xlabel('Days')
	#plt.ylabel('Adjusted Close')
	#plt.show()

	#cwd = os.getcwd()
	#cwd = cwd + '\\' + diagramFolderName
	#plt.savefig(cwd + '\\' + stock_name + diagramPictureName + '.png')
	#plt.close('all') # 关闭图 0
	return newp, newy_test


def calc_deviation(stock_name, lst_real, lst_predict, method):
	count = 0
	for index in range(len(lst_real)):
		deviation = round(float((lst_predict[index] - lst_real[index]) / lst_predict[index]),2)
		deviation = abs(deviation)
		count = count + deviation

	rate = round((count / len(lst_real)),2)

	stockPredictResultList = []
	stockPredictResult = {}
	stockPredictResult['StockNo'] = stock_name
	stockPredictResult['Type'] = method
	stockPredictResult['CorrectRate'] = rate
	stockPredictResultList.append(stockPredictResult)
	InsertStockPredictResult(stockPredictResultList)

def cal(lst_real, lst_predict, df, stock_name):
	keepFlag = 0
	totalBuyMoney = 0
	totalSellMoney = 0	
	stopValue = 0
	tmpCloseValue = 0
	count = 0

	cwd = os.getcwd()
	cwd = cwd + '\\' + outputFolderName + '\\'

	with open(cwd + stock_name + '_' + 'output.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['日期','動作','花費','收入','預期漲跌點數','實際漲跌點數','總花費','總收入'])
		for idx in range(len(lst_real) -2):
			msg = ''
			if keepFlag == 1:
				closeValue = round(float(lst_real[idx][0]),2)			

				if closeValue > tmpCloseValue:
					tmpCloseValue = closeValue
					stopValue = round(closeValue*0.9,2)

			#print(str(df.iloc[[4273+idx+1]].index[0]))
			date = str(df.iloc[[4273+idx+1]].index[0])
			msg = msg + str(df.iloc[[4273+idx+1]].index[0]) + ','
			expect = round((lst_predict[idx+1][0] - lst_real[idx][0]),2)
			real = round((lst_real[idx+1][0] - lst_real[idx][0]),2)
			if(keepFlag == 0 and (lst_predict[idx+1][0] - lst_real[idx][0]) >= 0):
				keepFlag = 1
				buyMoney = round(lst_real[idx+1][0],2)*1000
				totalBuyMoney = totalBuyMoney + buyMoney
				writer.writerow([date,'買進',str(buyMoney),'0',str(expect),str(real),str(totalBuyMoney),str(totalSellMoney)])
				#msg = msg + '買進,' + '花費:' + str(buyMoney) + ',' + '收入:0,' + '總花費:' + str(totalBuyMoney) + ',' + '總收入:' + str(totalSellMoney)
				#print('買進,' + '花費:' + str(buyMoney) + ',' + '收入:0,' + '總花費:' + str(totalBuyMoney) + ',' + '總收入:' + str(totalSellMoney))
			elif( idx == len(lst_real) -3 or (keepFlag == 1 and lst_predict[idx+1][0] < stopValue)):
				count = count + 1
				tmpCloseValue = 0
				keepFlag = 0
				sellMoney = round(lst_real[idx+1][0],2)*1000
				totalSellMoney = totalSellMoney + sellMoney
				writer.writerow([date,'賣出','0',str(sellMoney),str(expect),str(real),str(totalBuyMoney),str(totalSellMoney)])
				#msg = msg + '賣出,' + '花費:0,' + '收入:' + str(sellMoney) + ',' + '總花費:' + str(totalBuyMoney) + ',' + '總收入:' + str(totalSellMoney)
			
			elif keepFlag == 1:
				writer.writerow([date,'持有','0','0',str(expect),str(real),str(totalBuyMoney),str(totalSellMoney)])
				#msg = msg + '持有,' + '花費:0,' + '收入:0,' + '總花費:' + str(totalBuyMoney) + ',' + '總收入:' + str(totalSellMoney)
			
			else:
				writer.writerow([date,'無動作','0','0',str(expect),str(real),str(totalBuyMoney),str(totalSellMoney)])
				#msg = msg + '無動作,' + '花費:0,' + '收入:0,' + '總花費:' + str(totalBuyMoney) + ',' + '總收入:' + str(totalSellMoney)
		
			if(idx == len(lst_real) -3):
				print(msg)
				print(count)
				print()
	finalOutputFilePath = cwd + 'document.csv'
	with open(finalOutputFilePath,'a') as fd:
		writer = csv.writer(fd)
		writer.writerow([stock_name, str(totalBuyMoney), str(totalSellMoney), str(totalSellMoney-totalBuyMoney), str(round(((totalSellMoney-totalBuyMoney)/totalBuyMoney),2)), str(count)])

def calEveryUpDown(lst_real, lst_predict, df, stock_name):
	keepFlag = 0
	totalBuyMoney = 0
	totalSellMoney = 0	
	stopValue = 0
	tmpCloseValue = 0
	count = 0

	cwd = os.getcwd()
	cwd = cwd + '\\' + outputEveryUpDownFolderName + '\\'

	with open(cwd + stock_name + '_' + 'output.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['日期','動作','花費','收入','預期漲跌點數','實際漲跌點數','總花費','總收入'])
		for idx in range(len(lst_real) -2):
			msg = ''
			
			date = str(df.iloc[[4273+idx+1]].index[0])
			msg = msg + str(df.iloc[[4273+idx+1]].index[0]) + ','
			expect = round((lst_predict[idx+1][0] - lst_real[idx][0]),2)
			real = round((lst_real[idx+1][0] - lst_real[idx][0]),2)
			if(keepFlag == 0 and expect > 0):
				keepFlag = 1
				buyMoney = round(lst_real[idx+1][0],2)*1000
				totalBuyMoney = totalBuyMoney + buyMoney
				writer.writerow([date,'買進',str(buyMoney),'0',str(expect),str(real),str(totalBuyMoney),str(totalSellMoney)])
				
			elif(keepFlag == 1 and (idx == len(lst_real) -3 or expect < 0)):
				count = count + 1
				keepFlag = 0
				sellMoney = round(lst_real[idx+1][0],2)*1000
				totalSellMoney = totalSellMoney + sellMoney
				writer.writerow([date,'賣出','0',str(sellMoney),str(expect),str(real),str(totalBuyMoney),str(totalSellMoney)])
				
			if(idx == len(lst_real) -3):
				print(count)

	finalOutputFilePath = cwd + 'document.csv'
	with open(finalOutputFilePath,'a') as fd:
		writer = csv.writer(fd)
		writer.writerow([stock_name, str(totalBuyMoney), str(totalSellMoney), str(totalSellMoney-totalBuyMoney), str(round(((totalSellMoney-totalBuyMoney)/totalBuyMoney),2)), str(count)])

#stockList = list(GetStockList())

stockList = list(GetGoodPredictStock(0.02))

for i in range(len(stockList)):
	stock_name = str(stockList[i].get('StockNo'))
	print(stock_name)

	seq_len = 22
	try:	
		df = get_stock_data(stock_name, normalize=True)
		df_real = get_real_stock_data(stock_name, normalize=True)
		X_train, y_train, X_test, y_test = load_data(df, seq_len)

		cwd = os.getcwd()
		cwd = cwd + '\\' + modelFolderName
		tmpModelFileName = cwd + '\\' + stock_name + modelFileName + '.h5'
		model = load_model(tmpModelFileName)
		
		y_p = model_score(model, X_train, y_train, X_test, y_test)

		denorm_pred, denorm_real = plot_result(stock_name, y_p, y_test)

		calEveryUpDown(denorm_real, denorm_pred, df, stock_name)
		#calc_deviation(stock_name, list(denorm_pred), list(denorm_real), correctRateType)

		

		#cwd = os.getcwd()
		#cwd = cwd + '\\' + modelFolderName
		#model.save(cwd + '\\' + stock_name + modelFileName + '.h5')
	except Exception as e:
		print('error:' + stock_name + ' msg: ' + e)