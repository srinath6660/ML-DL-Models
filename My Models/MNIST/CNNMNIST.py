from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load the data and split it into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])

# Reshape the data to fit the model
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# One-hot-encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#print new label
print(y_train_one_hot[0])

# Build the CNN model
classifier = Sequential()
classifier.add(Convolution2D(64, kernel_size = 3,  input_shape = (28, 28, 1),
                             activation = 'relu'))
classifier.add(Convolution2D(32, kernel_size = 3, activation = 'relu'))
classifier.add(Flatten())
classifier.add(Dense(10, activation = 'softmax'))


# Comiple
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

# Train the model
hist = classifier.fit(X_train, y_train_one_hot, validation_data = (X_test, y_test_one_hot), 
               epochs = 3)

# Visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

# Show predictions as probabilities for the first 4 images in the test set
predictions = classifier.predict(X_test[:4])
print(predictions)

# Print our predictions as number labels
print(np.argmax(predictions, axis = 1))
# Print the actual labels
print(y_test[:4])

# Show the first 4 images as pictures
for i in range(0,4):
    image = X_test[i]
    image = np.array(image, dtype = 'float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap = 'gray')
    plt.show()