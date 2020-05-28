import tensorflow as tf 
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt 

#loading data
data = keras.datasets.fashion_mnist

(train_images,train_label),(test_images,test_label) = data.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#creating model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss=keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

#training model
model.fit(train_images,train_label,epochs=5)

#prediction
prediction = model.predict(test_images)

for i in range(5):
    plt.grid(True)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.xlabel("Actual:"+ class_names[test_label[i]])
    plt.title("Prediction:"+ class_names[np.argmax(prediction[i])])
    plt.show()

#to evaluate model
#print(model.evaluate(test_images,test_label)) 

#to see what our image looks like
#plt.imshow(test_images[1],cmap=plt.cm.binary) 
#plt.show()