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

# forme du réseau 2 neurone d'entrée 1 de sortie qui a pour f activation relu6
# Aucue couche profonde
# La fonction loss = Erreur quadratique moyenne (mean_squared_error)
#          _______
#   21 ---|       |
#         | relu6 | --- 30
#   9  ---|_______|
# 


model = CreateModel() 

# Train
model.fit(train_input,train_output,epochs=300,verbose=1,validation_split=1)
# Le chiffres loss correspond à la précision du model 
# Plus loss petit plus c'est précis

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