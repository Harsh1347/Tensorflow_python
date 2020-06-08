import math
import pandas as pd 
import numpy as np
import pandas_datareader as web 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , LSTM
from matplotlib import pyplot as plt 

plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL',data_source = 'yahoo',start = '2012-01-01',end = '2020-5-31')

# print(df.shape)

# plt.figure(figsize = (16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date',fontsize = 18)
# plt.ylabel('Close Price USD ($)',fontsize = 18)
# plt.show()

data2 = df.filter(['Close'])
print(data2.head())
data =  data2.values

train_data_len = round(len(data)*.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# print(scaled_data)

train_data = scaled_data[:train_data_len,:]
 
x_train = []
y_train = []

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

x_train , y_train = np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape = (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25,activation = 'relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(x_train,y_train,epochs=1,batch_size=1)

test_data = scaled_data[train_data_len-60:,:]

x_test =[]
y_test =data[train_data_len:,:]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)

rmse = np.sqrt(np.mean(prediction-y_test)**2)
print(rmse)

train = data2[:train_data_len]
valid = data2[train_data_len:]
valid['Predictions'] = prediction

plt.figure(figsize=(15,7))
plt.title('Model')
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['train','val','pred'])
plt.show()

