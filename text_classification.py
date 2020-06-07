import tensorflow as tf 
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt 

data = keras.datasets.imdb

(train_data,train_label),(test_data,test_label) = data.load_data(num_words=88000)

print(train_data[0])

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"]=0
word_index["<START>"]=1
word_index["<UNK>"]=2
word_index["<UNUSED>"]=3

reverse_word_index = dict([(val,key) for (key,val) in  word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding="post",maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding="post",maxlen=256)

def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])
#CREATING MODEL

model = keras.Sequential()
model.add(keras.layers.Embedding(88000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_label[:10000]
y_train = train_label[10000:]

fitmodel = model.fit(x_train,y_train,batch_size=512,epochs=40,verbose=1,validation_data=(x_val,y_val))

results = print(model.evaluate(test_data,test_label))
#SAVING MODEL

#model.save('data//text_clf.h5')
# review_data = test_data[0]
# predict = model.predict([review_data])
# print('review',decode_review(review_data))
# print("actual:",test_label[0])
# print("predicted:",predict[0])
# # print(decode_review(test_data[0]))

#USING A SAVED MODEL
def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    
    return encoded


model = keras.models.load_model('data//text_clf.h5')

with open ("data//test.txt") as f:
    for line in f.readlines():
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").replace("!","").replace("?","").strip()
        encode =  review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode],value=word_index['<PAD>'],padding="post",maxlen=256)
        predict=  model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])