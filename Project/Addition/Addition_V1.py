# Bilan : Ca marche pas MAIS ! 
# Cette algo permet de créer un reseau de neurone 
# Avec X sortie et le principe c'est de récupéré le numéro du neurone avec le plus grand nombre
# Cette approche fonction pas du tout pour une addition 
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
for i in range(3000-1):
    val1 = ran.randint(0,1000) 
    val2 = ran.randint(0,1000)
    val3 = val1+val2
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
    keras.layers.Dense(4, activation=tf.nn.relu6),
    keras.layers.Dense(2000, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# On arrète tout  ! le dataset ne vas pas : J'ai un nombre de sorti trop grande x+x
# Train
model.fit(ntrain_input,train_label,epochs=100,verbose=1)

# Validation

ntrain_test= np.array([[[99,741]]])
predict = model.predict(ntrain_test)

print('99+741 =', np.argmax(predict[0]),'840')

#Prediction