'''
from MongoDB import *
import csv

data = GetStockInfoList('1101')
dataList = list(data)
# Date,Open,High,Low,Last,Close,Total Trade Quantity,Turnover (Lacs)

with open('train.csv', 'w', newline='') as csvfile:
	# 建立 CSV 檔寫入器
	writer = csv.writer(csvfile)

	# 寫入一列資料
	writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Quantity', 'PERatio'])

	# 寫入另外幾列資料
	for idx in range(len(dataList)):
		realIdx = len(dataList) - idx -1
		try:
			writer.writerow([dataList[realIdx].get('Date'),
					dataList[realIdx].get('StockInfo').get('stockOpenList'),
					dataList[realIdx].get('StockInfo').get('stockHighList'),
					dataList[realIdx].get('StockInfo').get('stockLowList'),
					dataList[realIdx].get('StockInfo').get('stockCloseList'),
					dataList[realIdx].get('StockInfo').get('stockTranMountList').replace(',',''),
					dataList[realIdx].get('StockInfo').get('StockPERatio')])
		except:
			print(realIdx)
'''

def correctRate(lstReal, lstPredict, stockNo):
	count = 0;
	for idx in range(1,len(lstReal)):
		realUpOrDown = lstReal[idx] - lstReal[idx-1]
		realBoolUp = True
		if realUpOrDown >= 0:
			realBoolUp = True
		else:
			realBoolUp = False

		predictUpOrDown = lstPredict[idx] - lstPredict[idx-1]
		predictBoolUp = True
		if predictUpOrDown >= 0:
			predictBoolUp = True
		else:
			predictBoolUp = False

		if realBoolUp == predictBoolUp:
			count = count +1

	print(stockNo + ':' + str(float(count/len(lstReal))))


# Import the libraries
import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
import pandas as pd

# Import the training set
dataset_train = pd.read_csv('train.csv')  # 讀取訓練集
training_set = dataset_train.iloc[:, 1:2].values  # 取「Open」欄位值

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []   #預測點的前 60 天的資料
y_train = []   #預測點
for i in range(60, 1676):  # 1258 是訓練集總數
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 進行訓練
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

dataset_test = pd.read_csv('test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # Feature Scaling

X_test = []
for i in range(60, 341):  # timesteps一樣60； 80 = 先前的60天資料+2017年的20天資料
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape 成 3-dimension

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)  # to get the original scale

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real 1101 Stock Price')  # 紅線表示真實股價
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted 1101 Stock Price')  # 藍線表示預測股價
plt.title('1101 Prediction')
plt.xlabel('Time')
plt.ylabel('1101 Price')
plt.legend()
plt.show()

#regressor.save('1101_mi=odel.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
#del model  # deletes the existing model

