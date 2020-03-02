# La même que le 3 + des threads lol
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import threading
def CreateModel(a):
    
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(a, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model


# Main

# model = CreateModel(128)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# tab = [152,158,115,93,170,154,117,165,110,135]
tab = [154,115]
for j in range(0,10):
    for i in tab:
        model = CreateModel(i)
        model.fit(train_images,train_labels,epochs=5,verbose=0)
        test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=0)
        print(str(i)+";"+str(test_acc))
