import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

data = pd.read_csv("stock_data.csv")

print(data.head())

print(data['rpt_key'].value_counts())

df = data.loc[(data['rpt_key'] == 'btc_usd')]

print(df.head())

df = df[['last']]
dataset = df.values
dataset = dataset.astype('float32')
#print(dataset)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#print(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
#print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 10
trainX, trainY = create_dataset(train, look_back=look_back)
testX, testY = create_dataset(test, look_back=look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#BUILD MODEL
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=256, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

print('ACTUAL '+str(len(trainY[0])))
print(trainY[0])
print('PREDICT '+str(len(trainPredict[:, 0])))
print(trainPredict[:, 0])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Error Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Error Score: %.2f RMSE' % (testScore))

testY_plot = testY[0]
testPredict_plot = testPredict[:, 0]

plt.plot( testY_plot, color='red', marker='o', label='ACTUAL', linestyle = 'None')
plt.plot( testPredict_plot, color='blue', marker='X', label='PREDICT', linestyle = 'None')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('RNN')
plt.legend()
plt.show()
