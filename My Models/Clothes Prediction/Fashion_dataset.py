import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

dataset = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labesl) = dataset.load_data()

img_index = 3
img = train_images[img_index]
print("Image label is:", train_labels[img_index])
plt.imshow(img)

# Creating ANN
# Lets make ANN
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Flatten

classifier = Sequential()
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(10, activation = 'softmax'))

# Compile
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', 
                   metrics = ['accuracy'])   

# Fitting the ANN to the training set
classifier.fit(train_images, train_labels, batch_size = 32, epochs = 5)     

#Evaluate the Model
classifier.evaluate(test_images, test_labesl)

predictions = classifier.predict(test_images[0:5]) 

print(predictions)

print(np.argmax(predictions, axis = 1))
print(test_labesl[0:5]) # ORIGINAL

for i in range(0,5):
    plt.imshow(test_images[i], cmap = 'gray')
    plt.show()

                   