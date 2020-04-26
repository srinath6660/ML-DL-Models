import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

X = mnist['data']
y = mnist['target']

X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]


'''# Flattening the images
X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))
y_train = y_train.reshape((-1, 784))
y_test = y_test.reshape((-1, 784))'''

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
# Building the model
classifier = Sequential()
classifier.add(Dense(64, activation = 'relu', input_dim = 784))
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(10, activation = 'softmax'))

# Compile the model
# the loss function measures how well the model did 
#on training and then tries to imporve on it using optimizer

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

# Train the model
classifier.fit(X_train, y_train, batch_size = 32, epochs = 5)