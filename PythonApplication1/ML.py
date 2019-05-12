'''
from MongoDB import *
import pandas as pd
import pandas_datareader.data as web
import os
import datetime

stockList = list(GetStockList())
for i in range(len(stockList)):
	stock_name = stockList[i].get('No') + '.TW'
	start = datetime.datetime(1990, 1, 1)
	end = datetime.date.today()
	df = web.DataReader(stock_name, "yahoo", start, end)
	cwd = os.getcwd()
	cwd = cwd + '\\dataframe\\' + stock_name
	df.to_pickle(cwd)
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math
import time
import itertools
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import pandas_datareader.data as web
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from MongoDB import *
from keras.callbacks import EarlyStopping

dataframeFolderName = 'dataframe'
diagramFolderName = 'lstm_train_diagram_100'
diagramPictureName = '_LSTM_prediction_diagram'
modelFolderName = 'lstm_train_model_100'
modelFileName = '_LSTM_prediction'
correctRateType = '5_features_lstm_predict_100'

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

def load_data(stock, seq_len):
	amount_of_features = len(stock.columns)
	data = stock.as_matrix() 
	sequence_length = seq_len + 1 # index starting from 0
	result = []

	for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
		result.append(data[index: index + sequence_length]) # index : index + 22days

	result = np.array(result)
	row = round(0.9 * result.shape[0]) # 90% split

	train = result[:int(row), :] # 90% date
	X_train = train[:, :-1] # all data until day m
	y_train = train[:, -1][:,-1] # day m + 1 adjusted close price

	X_test = result[int(row):, :-1]
	y_test = result[int(row):, -1][:,-1] 

	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

	return [X_train, y_train, X_test, y_test]

def build_model2(layers, neurons, d):
	model = Sequential()

	model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
	model.add(Dropout(d))

	model.add(LSTM(neurons[1], input_shape=(layers[1], layers[2]), return_sequences=False))
	model.add(Dropout(d))

	model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
	model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))
	# model = load_model('my_LSTM_stock_model1000.h5')
	adam = keras.optimizers.Adam(decay=0.2)
	model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model

def model_score(model, X_train, y_train, X_test, y_test):
#其實這裡只需要傳入X_test和y_test即可，只是為了和上面格式保持一致因而也將X_train和y_train傳入了
	y_hat = model.predict(X_test)
	y_t = y_test.reshape(-1,1)

	temp = pd.DataFrame(y_hat)  
	temp['yhat'] = y_hat
	temp['y'] = y_t
	temp_rmse = sqrt(mean_squared_error(temp.y,temp.yhat))
	temp_mse = mean_squared_error(temp.y,temp.yhat)
	print('TEMP RMSE: %.3f' % temp_rmse)
	print('TEMP MSE: %.3f' % temp_mse)
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
	fig = plt.figure()
	plt.plot(newp, color='red', label='Prediction')
	plt.plot(newy_test,color='blue', label='Actual')
	plt.legend(loc='best')
	plt.title('The test result for {}'.format(stock_name))
	plt.xlabel('Days')
	plt.ylabel('Adjusted Close')
	#plt.show()

	cwd = os.getcwd()
	cwd = cwd + '\\' + diagramFolderName
	plt.savefig(cwd + '\\' + stock_name + diagramPictureName + '.png')
	plt.close('all') # 关闭图 0
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

#stockList = list(GetStockList())
goodStockList = list(GetGoodPredictStock(1.00))
stockList = [1101,
1102,
1216,
1301,
1303,
1326,
1402,
1476,
2002,
2105,
2207,
2301,
2303,
2308,
2317,
2324,
2330,
2354,
2357,
2382,
2395,
2408,
2409,
2412,
2474,
2801,
2880,
2881,
2882,
2883,
2884,
2885,
2886,
2311,
2325,
2887,
2890,
2891,
2892,
2912,
3008,
3045,
3474,
3481,
4904,
4938,
5880,
6505,
9904,
1227,
1262,
1319,
1440,
1477,
1504,
1536,
1589,
1590,
1605,
1704,
1707,
1717,
1722,
1723,
1789,
1802,
2015,
2049,
2059,
2101,
2103,
2106,
2201,
2204,
2206,
2231,
2313,
2327,
2345,
2347,
2352,
2353,
2355,
2356,
2360,
2362,
2376,
2377,
2379,
2383,
2385,
2392,
2448,
2449,
2451,
2498,
2501,
2542,
2603,
2609,
2610,
2615,
2618,
2707,
2809,
2812,
2823,
2834,
2845,
2849,
2867,
2888,
2903,
2915,
3034,
3037,
3044,
3189,
3231,
3673,
3682,
3702,
4958,
5264,
5522,
5871,
6176,
6239,
6269,
6285,
6414,
6415,
6452,
6456,
8150,
8454,
8464,
9907,
9910,
9914,
9917,
9921,
9933,
9938,
9945]



for i in range(len(stockList)):
	stock_name = str(stockList[i]) + '.TW'

	try:
		obj = next(item for item in goodStockList if ((item["StockNo"] == stock_name) and (item["Type"] == "5_features_lstm_predict_100")))
		print("Pass " + str(stock_name))
		continue
	except:
		print("Do " + str(stock_name))	

	#print(stock_name)
	try:		
		seq_len = 22
		d = 0.2
		shape = [5, seq_len, 1] # feature, window, output
		neurons = [128, 128, 32, 1]
		epochs = 300

		df = get_stock_data(stock_name, normalize=True)

		X_train, y_train, X_test, y_test = load_data(df, seq_len)

		model = build_model2(shape, neurons, d)

		early_stopping = EarlyStopping(monitor='loss', patience=40, verbose=1)

		history = model.fit(X_train,
			y_train,
			batch_size=512,
			epochs=epochs,
			validation_split=0.1,
			verbose=1,
			callbacks=[early_stopping])

		y_p = model_score(model, X_train, y_train, X_test, y_test)

		denorm_pred, denorm_real = plot_result(stock_name, y_p, y_test)

		calc_deviation(stock_name, list(denorm_pred), list(denorm_real), correctRateType)

		#model = load_model('test.h5')

		cwd = os.getcwd()
		cwd = cwd + '\\' + modelFolderName
		model.save(cwd + '\\' + stock_name + modelFileName + '.h5')
	except:
		print(stock_name)