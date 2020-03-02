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


# ca c'est le thread
def thread_function():
    model = CreateModel(152)
    model.fit(train_images,train_labels,epochs=5,verbose=0)
    test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=0)
    print(str(model.layers[1].units)+";"+str(test_acc))


# Main

# model = CreateModel(128)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

for i in range(0,2):
    threading.Thread(target=thread_function).start()


