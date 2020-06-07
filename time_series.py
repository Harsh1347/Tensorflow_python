import pandas as pd
import numpy as np 
from sklearn import preprocessing
from collections import deque
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Dropout,LSTM,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint


SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10
BATCH_SIZE =64
NAME = f"{SEQ_LEN}-SEQ={FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(cur,fut):
    if float(fut) > float(cur):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop('Future',1)
    for col in df.columns:
        if col != "Target":
            df[col] = df[col].pct_change()
            df.dropna(inplace =True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace = True)

    seq_data =[]
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            seq_data.append([np.array(prev_days),i[-1]])

    random.shuffle(seq_data)

    buys = []
    sells = []

    for seq , target in seq_data:
        if target == 0:
            sells.append([seq,target])
        elif target ==1:
            buys.append([seq,target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys),len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    seq_data = buys+sells
    random.shuffle(seq_data)

    x=[]
    y=[]

    for seq,target in seq_data:
        x.append(seq)
        y.append(target)

    return np.array(x),y

# data = pd.read_csv("data//crypto_data//LTC-USD.csv",names = ['Time','Low','High','Open','Close','Volume'])
# main_df = pd.DataFrame()
# ratio = ['BTC-USD','LTC-USD','ETH-USD','BCH-USD']
# for r in ratio :
#     data = f"data//crypto_data//{r}.csv"
#     df = pd.read_csv(data,names = ['Time','Low','High','Open','Close','Volume'])
#     df.rename(columns = {'Close':f'{r}_Close','Volume':f'{r}_Volume'},inplace = True)
#     df.set_index("Time",inplace = True)
#     df = df[[f'{r}_Close',f'{r}_Volume']]
#     if len(main_df) == 0:
#         main_df=df
#     else:
#         main_df = main_df.join(df)

# main_df.to_csv("data//crypto_data//combined.csv")
df = pd.read_csv('data//crypto_data//combined.csv')
df['Future'] = df[f'{RATIO_TO_PREDICT}_Close'].shift(-FUTURE_PERIOD_PREDICT)


df['Target'] = list(map(classify,df[f'{RATIO_TO_PREDICT}_Close'],df['Future']))

# print(df.head())

times = sorted(df.index.values)
last_5 = times[-int(0.05*len(times))]

validation_main_df = df[(df.index >= last_5)]
main_df = df[(df.index < last_5)]
# print(len(main_df),len(validation_main_df))

train_x,train_y= preprocess_df(main_df)
val_x,val_y = preprocess_df(validation_main_df)


print(len(train_x),len(val_x))
print(train_y.count(0),train_y.count(1))
print(val_y.count(0),val_y.count(1))

model = Sequential()
model.add(LSTM(128,input_shape =(train_x.shape[1:]),return_sequences = True,activation='tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape =(train_x.shape[1:]),return_sequences = True,activation='tanh'))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape =(train_x.shape[1:]),activation='tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2,activation= 'softmax'))

opt = tf.keras.optimizers.Adam(lr = 0.001,decay = 1e-6)

model.compile(loss = 'sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
            )

history = model.fit(
    np.array(train_x),np.array(train_y),
    batch_size=BATCH_SIZE,
    epochs = EPOCHS,
    validation_data=(np.array(val_x),np.array(val_y))
)

model.save("save.h5")