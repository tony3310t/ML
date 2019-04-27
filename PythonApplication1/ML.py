import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


def readTrain():
	train = pd.read_csv("NSE-TATAGLOBAL.csv")
	train = train.iloc[::-1]
	return train

def normalize(train):
	train = train.drop(["Date"], axis=1)
	train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
	return train_norm

def denormalize(train):
	#train = train.drop(["Date"], axis=1)
	train_norm = train.apply(lambda x: x * (np.max(x) - np.min(x)) + np.mean(x))
	return train_norm

def buildTrain(train, pastDay=30, futureDay=5):
	X_train, Y_train = [], []
	for i in range(train.shape[0] - futureDay - pastDay):
		X_train.append(np.array(train.iloc[i:i + pastDay]))
		#print(X_train)
		Y_train.append(np.array(train.iloc[i + pastDay:i + pastDay + futureDay]["Close"]))
		#print(Y_train)
	return np.array(X_train), np.array(Y_train)

def splitData(X,Y,rate):
	X_train = X[int(X.shape[0] * rate):]
	Y_train = Y[int(Y.shape[0] * rate):]
	X_val = X[:int(X.shape[0] * rate)]
	Y_val = Y[:int(Y.shape[0] * rate)]
	return X_train, Y_train, X_val, Y_val

def buildManyToOneModel(shape):
	model = Sequential()
	model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
	# output shape: (1, 1)
	model.add(Dense(1))
	model.compile(loss="mse", optimizer="adam")
	model.summary()
	return model

if __name__ == "__main__":
	train = readTrain()
	# train_Aug = augFeatures(train)
	train_norm = normalize(train)
	#train_denorm = denormalize(train_norm)
	# change the last day and next day
	X_train, Y_train = buildTrain(train_norm, 20, 1)
	#X_train, Y_train = shuffle(X_train, Y_train)
	# because no return sequence, Y_train and Y_val shape must be 2 dimension
	X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

	model = buildManyToOneModel(X_train.shape)
	callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
	model.fit(X_train, Y_train, epochs=100, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
	X_val = model.predict(X_train)


