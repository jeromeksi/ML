import numpy as np
import random as ran

def findTuple(ntrain_input,val1,val2):
    for item in ntrain_input[0] :
        if (item[0] == val2 and item[1] == val2) or (item[1] == val2 and item[0] == val2) :
            return True
    return False

# Cette fonction créer un dataset input et output
def CreateDataSet(size):
    train_input = np.array([[[0,0]]])
    train_output = [0]
    for i in range(size-1):
        a = ran.randint(0,1000)/10000.0
        b = ran.randint(0,1000)/10000.0
        tot = (a+b)
        if not findTuple(train_input,a,b):
            train_input = np.concatenate((train_input,[np.array([[a,b]])]))
            train_output.append(tot)
        else : 
            i-=1
    return train_input,train_output
# Cette fonction crére un dataset
def CreateModel():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1 ,2)),
        keras.layers.Dense(1, activation=tf.nn.relu6)
    ])
    model.compile(optimizer='Adam',
                loss=keras.losses.mean_squared_error)
    return model