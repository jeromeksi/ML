# Version final 
# Création d'un NN qui permet d'addition 2 nombre a et b (a,b ∈ {0,99})

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random as ran
from Helper.Function_Helper import CreateDataSet,CreateModel

# Créer mes deux dataset

train_input,train_output = CreateDataSet(1000)

# print(len(train_input))
# print(len(train_output))
# n= ran.randint(0,len(train_input))
# print(train_input[n])
# print(train_output[n])


# Créer NN 

model = CreateModel()

# Train
model.fit(train_input,train_output,epochs=300,verbose=1,validation_split=1)

# Validation | Prediction

ntrain_test= np.array([[[0.0099,0.0741]],
                        [[0.0099,0.0740]],
                        [[0.0099,0.0001]],
                        [[0.0999,0.0999]],
                        [[0.0000,0.0000]],
                        ])

predict = model.predict(ntrain_test)

print('99+741 =',int(round(predict[0][0]*10000,0)),'840')
print('99+740 =',int(round(predict[1][0]*10000,0)),'839')
print('99+1 =',int(round(predict[2][0]*10000,0)),'100')
print('999+999 =',int(round(predict[3][0]*10000,0)),'1998')
print('0+0 =',int(round(predict[4][0]*10000,0)),'0')