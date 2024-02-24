import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.datasets import mnist
from tensorflow.keras import regularizers

# Carga de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocesamiento de datos
X_train = x_train[0:50000] / 255 
Y_train = keras.utils.to_categorical(y_train[0:50000], 10) 

X_val = x_train[50000:60000] / 255
Y_val = keras.utils.to_categorical(y_train[50000:60000], 10)

X_test = x_test / 255
Y_test = keras.utils.to_categorical(y_test, 10)

# Construcción del modelo
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(600, activation='sigmoid', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(500, activation='sigmoid', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(300, activation='sigmoid', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(100, activation='sigmoid', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(10, activation='softmax'))

# Compilación del modelo con el optimizador Adam
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(X_train, Y_train, 
                    batch_size=128, 
                    epochs=40,
                    verbose=1, 
                    validation_data=(X_val, Y_val))

# Evaluación del modelo en datos de prueba
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f'Loss on test data: {loss:.4f}')
print(f'Accuracy on test data: {accuracy:.4f}')