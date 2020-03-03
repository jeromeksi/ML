# Import TF Num
import tensorflow as tf
import numpy as np

#data set 
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

#create layer
IO = tf.keras.layers.Dense(units=1,input_shape=[1])

#model
model = tf.keras.Sequential([IO])

#Compilation
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
#Training / Fit
model.fit(celsius_q,fahrenheit_a,epochs=1000,verbose=False)

#Predict
print(model.predict([100]))
print(IO.get_weights())