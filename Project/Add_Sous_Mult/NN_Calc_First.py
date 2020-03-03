# le principe est de faire un nn qui prend 3 entrée
# Le premier input correspond à l'opération a appliquer
# Les 2 autres correspond au nombre de l'opération 
# Exemple : [1,2,3] => 2+3 = 5
# 1 : +
# 2 : -
# 3 : x
# Ca marche je n'arrive pas a trouvé la bonne forme de nn, du coup 

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random as ran
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_epoch = 150

#Function
CreateModel.testF()
def Operate(valO,val1,val2):
    if(valO == 1):
        return val1+val2
    elif(valO == 2):
        return val1-val2
    elif(valO == 3):
        return val1*val2

#Création des input/output


ntrain_input = np.array([[[0,0,0]]])
train_label = [0]



for i in range(1000-1):
    val1 = ran.randint(0,100)
    val2 = ran.randint(0,100)
    valO = ran.randint(1,3)

    val3 = Operate(valO,val1,val2)/10000.0
    
    var = [val1,val2]
    ntrain_input = np.concatenate((ntrain_input,[np.array([[valO/10000.0,val1/10000.0,val2/10000.0]])]))
    train_label.append(val3)


# print(len(ntrain_input))
# print(len(train_label))
# print(ntrain_input[0])
# print(train_label[0])

print(ntrain_input[1])
print(train_label[1])

print(ntrain_input[2])
print(train_label[2])


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1 ,3)),
    keras.layers.Dense(1, activation=tf.nn.tanh)
])
model.compile(optimizer='Adam',
              loss=keras.losses.mean_squared_error)

# Train
print('\n\n\nStart trainning...')
loss = model.fit(ntrain_input,train_label,epochs=n_epoch,verbose=0)
print('loss=',loss.history['loss'][n_epoch-1])

#test
ntrain_test= np.array([ [[1,0.0099,0.0071]],
                        [[3,0.0099,0.0074]],
                        [[2,0.0099,0.0001]],
                        [[1,0.0099,0.0099]],
                        [[2,0.0050,0.0010]],
                        ])

predict = model.predict(ntrain_test)

print('99+71 =',int(round(predict[0][0]*10000,0)),'170')
print('99+71 =',predict[0][0],'170')
print('99*74 =',int(round(predict[1][0]*10000,0)),'7326')
print('99-1 =',int(round(predict[2][0]*10000,0)),'98')
print('99+99 =',int(round(predict[3][0]*10000,0)),'198')
print('50-10 =',int(round(predict[4][0]*10000,0)),'40')

