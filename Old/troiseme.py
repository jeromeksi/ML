# On fait la même que le premier
# mais le but c'est de créer une liste de model 
# ongénère les models de manière "aléatoire"
# On commence par changer le nombre de neurones dans la couche
# Puis on véra si on change d'autre choses

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import time

# Helper libraries
import numpy as np

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
# j=0
# tab = []
# for i in range(1,200):
#     start_time = time.time()
#     tab.append([CreateModel(152),0])
#     tab[j][0].fit(train_images,train_labels,epochs=5,verbose=0)
#     test_loss, test_acc = tab[j][0].evaluate(test_images,test_labels,verbose=0)
#     tab[j][1] = test_acc
#     print(str(tab[j][0].layers[1].units)+";"+str(tab[j][1]))
#     j+=1

# print('')

for i in range(1,200):
    model = CreateModel(152)
    model.fit(train_images,train_labels,epochs=5,verbose=0)
    test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=0)
    print(str(model.layers[1].units)+";"+str(test_acc))
