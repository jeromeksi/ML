# Pri,ncipe faire des nouveaux test sur l'addition 
# Addition d'entier négatif et positif pour la V1
# La version final vise l'addition de 2 nombres dans l'ensemble R
from Helper.Function_Helper import CreateDataSet,CreateModel
import numpy as np
import tensorflow as tf

i_size = 100
i_epoch = 2000
f_normalize = 1.0
# Dataset
train_input,train_output = CreateDataSet(i_size)
train_input  = train_input / f_normalize
train_output = train_output / f_normalize
# Model
forme_nn = np.array([[None,None,(2,),tf.keras.layers.Flatten]]) #Input
# forme_nn = np.concatenate((forme_nn,np.array([[4,tf.nn.relu,None,tf.keras.layers.Dense]])))
forme_nn = np.concatenate((forme_nn,np.array([[1,tf.nn.elu,None,tf.keras.layers.Dense]]))) #Output

model = CreateModel(forme_nn)

# Training
model.fit(train_input,train_output,epochs=i_epoch,verbose=1)

# Validation

train_test= np.array([[99,741],
                        [99,740],
                        [99,1],
                        [-999,999],
                        [0,0]
                        ])
train_test =train_test / f_normalize          
predict = model.predict(train_test)

print('99+741 | Prédiction =',int(round(predict[0][0]*f_normalize,0)),'| Résultat = 840')
print('99+740 | Prédiction =',int(round(predict[1][0]*f_normalize,0)),'| Résultat = 839')
print('99+1 | Prédiction =',int(round(predict[2][0]*f_normalize,0)),'| Résultat = 100')
print('999+999 | Prédiction =',int(round(predict[3][0]*f_normalize,0)),'| Résultat = 0')
print('0+0 | Prédiction =',int(round(predict[4][0]*f_normalize,0)),'| Résultat = 0')