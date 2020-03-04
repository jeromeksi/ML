# Bilan : Ca fonctionne !! 
# Le principe : 2 neurone ne entrée et 1 en sortie 
# le neruone de sortie s'active sur une fonctionne sigmoid
# le max possible dans cette exemple  999+999
# Cette version fonction correctement MAIS 2 petits problème
#   1. Très mal structuré (c'est nomal c'est la version final)
#   2. La gestion des tableaux est mauvais, les tableaux d'input 
#       sont sous a forme [[[w,x]],[[y,z]]] le mieux est => [[w,x],[y,z]]

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random as ran

# Trouver un tuple dans un tableau

def findTuple(ntrain_input,val1,val2):
    for item in ntrain_input[0] :
        if (item[0] == val2 and item[1] == val2) or (item[1] == val2 and item[0] == val2) :
            return True
    return False


# Créer mes deux dataset

ntrain_input = np.array([[[0,0]]])
# train_input = [[0,0]]
train_label = [0]
i =0
for i in range(1000-1):
    val1 = ran.randint(0,1000)/10000.0
    val2 = ran.randint(0,1000)/10000.0
    val3 = (val1+val2)
    if not findTuple(ntrain_input,val1,val2) :
        var = [val1,val2]
        ntrain_input = np.concatenate((ntrain_input,[np.array([[val1,val2]])]))
        train_label.append(val3)

print(len(ntrain_input))
print(len(train_label))
n= ran.randint(0,len(ntrain_input))
print(ntrain_input[n])
print(train_label[n])


# Créer NN 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1 ,2)),
    keras.layers.Dense(1, activation=tf.nn.relu6)
])
model.compile(optimizer='Adam',
              loss=keras.losses.mean_squared_error)

# Train
model.fit(ntrain_input,train_label,epochs=300,verbose=1,validation_split=1)

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