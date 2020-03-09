#Soustraction de 2 chiffres
# 2 nombres en entrée
# 1 nombre en sortie compris entres -1 et 1
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random as ran
from Helper.Function_Helper import CreateDataSet,CreateModel

# Variables
n_epochs = 100
n_size_Dataset = 1000
n_normalize = 10000.0


# DataSet 
train_input,train_output = CreateDataSet(n_size_Dataset)

print(train_input[2])
print(train_output[2])


train_input = train_input/n_normalize
train_output = train_output / n_normalize

print(train_input[2])
print(train_output[2])



# Model 
# TODO : Créer un objet NN qui contiendra les informations
# Un objet Input et un Output,un tableau d'objet Layer pour les couches   
forme_nn = np.array([[None,None,(2,),keras.layers.Flatten]]) #Input
forme_nn = np.concatenate((forme_nn,np.array([[4,tf.math.sin,None,keras.layers.Dense]])))
forme_nn = np.concatenate((forme_nn,np.array([[1,tf.math.sin,None,keras.layers.Dense]]))) #Output

model = CreateModel(forme_nn)

# Training
model.fit(train_input,train_output,epochs=n_epochs, verbose=1,validation_split=0)

# DataSet Prédiction 

data_validate = np.array([
                [45,5],
                [5,10],
                [42,42],
                [42,10],
                [20,910],
                [587,153]
                ])
data_validate = data_validate / n_normalize
# Prediction
predict = model.predict(data_validate)

# Résultat 
print('45-5 Prédiction =',int(round(predict[0][0]*10000,0)),'| Résultat =40')
print('5-10 Prédiction =',int(round(predict[1][0]*10000,0)),'| Résultat =-5')
print('42-42 Prédiction =',int(round(predict[2][0]*10000,0)),'| Résultat =0')
print('42-10 Prédiction =',int(round(predict[3][0]*10000,0)),'| Résultat =-32')
print('20-910) Prédiction =',int(round(predict[4][0]*10000,0)),'| Résultat =-890')
print('587-153 Prédiction =',int(round(predict[5][0]*10000,0)),'| Résultat =434')