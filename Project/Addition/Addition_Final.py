# Version final 
# Création d'un NN qui permet d'addition 2 nombre a et b (a,b ∈ {0,999})
# Note : Il existe de rare cas où le réseau reste bloqué 
#        avec un loss trop élevé pour être utilisable...
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random as ran
from Helper.Function_Helper import CreateDataSet,CreateModel

# Variables
n_epochs = 300
n_size_Dataset = 1000
n_normalize = 10000.0
# Créer mes deux dataset

train_input,train_output = CreateDataSet(n_size_Dataset)

train_input = train_input/n_normalize
train_output = train_output / n_normalize
# Normalisation consiste a créer des nombres entre 0 et 1 
# Exemple =  621 => 0.0621 


# Créer NN 

# forme du réseau : 2 neurones d'entré et 1 de sortie qui a pour f activation relu6
# Aucue couche profonde
# Fonction objectif (loss function) = Erreur quadratique moyenne (mean_squared_error)
#          _______
#   21 ---|       |
#         | relu6 | --- 30
#   9  ---|_______|
# Thérorie
# Le principe du réseau : s = relu6(x*W1 + y*W2)
# 
#       W1  _______
#   x -----|       |
#          | relu6 | --- s
#   y -----|_______|
#       W2
#
# W1 et W2 sont les poids appliqués au lien de chaque neurones
# Dans cette exemple simpliste le W1 et W2 doivent être égale à 1

model = CreateModel() 
# La fonction CreateModel n'a pas parametrable pour ce projet

# Train
model.fit(train_input,train_output,epochs=n_epochs,verbose=1,validation_split=1)
# Le chiffres loss correspond à la précision du model 
# Plus loss petit plus c'est précis
# Loss = l'écart entre la valeur prédit et la valeur voulu
# epochs = le nombre de fois ou le dataset va être utilisé pour l'apprentissage
# verbose = 1) affiche des informationss | 0) pas d'affichage
# validation_split = 1) utilise une partie du dataset d'input pour la validation 0) pas d'utilsiation
# validation_split = je suis pas sur de l'impact de cette option


# Validation | Prediction

# ntrain_test= np.array([[[99,741]],
#                         [[99,740]],
#                         [[99,1]],
#                         [[999,999]],
#                         [[0,0]],
#                         ])
ntrain_test= np.array([[99,741],
                        [99,740],
                        [99,1],
                        [999,999],
                        [0,0]
                        ])
ntrain_test =ntrain_test / n_normalize          
predict = model.predict(ntrain_test)

print('99+741 | Prédiction =',int(round(predict[0][0]*10000,0)),'| Résultat = 840')
print('99+740 | Prédiction =',int(round(predict[1][0]*10000,0)),'| Résultat = 839')
print('99+1 | Prédiction =',int(round(predict[2][0]*10000,0)),'| Résultat = 100')
print('999+999 | Prédiction =',int(round(predict[3][0]*10000,0)),'| Résultat = 1998')
print('0+0 | Prédiction =',int(round(predict[4][0]*10000,0)),'| Résultat = 0')