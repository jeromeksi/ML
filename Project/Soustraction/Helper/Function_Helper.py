import numpy as np
import random as ran
import tensorflow as tf
from tensorflow import keras
def findTuple(ntrain_input,val1,val2):
    for item in ntrain_input :
        if (item[0] == val2 and item[1] == val2) or (item[1] == val2 and item[0] == val2) :
            return True
    return False

# Cette fonction créer un dataset input et output
def CreateDataSet(size):
    train_input = np.array([[0,0]])
    train_output = np.array(0)
    for i in range(size-1):
        a = ran.randint(0,1000)
        b = ran.randint(0,1000)
        tot = (a-b)
        if not findTuple(train_input,a,b):
            train_input = np.concatenate((train_input,np.array([[a,b]])))
            train_output = np.append(train_output,tot)
        else : 
            i-=1
    return train_input,train_output


# Cette fonction crére un model 
def CreateModel(forme):
    model = keras.Sequential()
    
    for item in forme:
        if(item[3]==keras.layers.Flatten):
            model.add(keras.layers.Flatten(input_shape=item[2]))
        elif(item[3]==keras.layers.Dense):
            model.add(keras.layers.Dense(item[0], activation=item[1]))

    model.compile(optimizer='Adam',
                loss=keras.losses.mean_squared_error) 
    return model