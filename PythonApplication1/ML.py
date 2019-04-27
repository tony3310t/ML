import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Part 1- Data Preprocessing
#importing training set
training_set=pd.read_csv('NSE-TATAGLOBAL.csv')
#extract open value from the trainng data
training_set=training_set.iloc[:,1:2].values
#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set=sc.fit_transform(training_set)
#Getting the input and output
X_train= training_set[:1236]
print(X_train)
Y_train=training_set[1:1237]
print(Y_train)
#Reshaping
X_train=np.reshape(X_train,(1236,1,1))
#Part-2 Building RNN
#importing keras library and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#Initalizing RNN
regressor=Sequential()
regressor.add(LSTM(units=50,activation='sigmoid', input_shape=(1,1)))
#Adding output layer (default argument)
regressor.add(Dense(units=1))
#Compile LSTM
regressor.compile(optimizer='adam',loss='mean_squared_error')
#Fitting the RNN on training set
regressor.fit(X_train,Y_train,batch_size=50,epochs=200)
#Part 3-Making Prediction and Visualizing Results
#Getting real Stock price for 2017
test_set=pd.read_csv('tatatest.csv')
real_stock_price=test_set.iloc[:,1:2].values
print(real_stock_price)
real_stock_price1=test_set.iloc[:,1:2].values
print(real_stock_price1)
#Getting predicted Stock price for 2017
inputs=real_stock_price
inputs=sc.transform(inputs)
inputs=np.reshape(inputs,(16,1,1))  #scaling the values
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #scaling to input values
#Visualize the results
x=[]
y1=[]
for i  in range(16):
    x.append(i)
for j in range(16):
    y1.append(j)

print(real_stock_price1)
print('------------------------')
print(predicted_stock_price)
plt.plot(x,real_stock_price1,'ro',color='red',label='Real Stock Price')
plt.plot(y1,predicted_stock_price,'ro',color='green',label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()