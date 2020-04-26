# CLASSIFY IMAGES

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

import matplotlib.pyplot as plt
img  = plt.imshow(X_train[0])
y_train[0]

# One-Hot-Encoding
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# print new labels
print(y_train_one_hot)

# Normalize the pixels in the images to be between 0 and 1
X_train = X_train/255
X_test = X_test/255

# Buliding CNN
# Building the CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step1 : Convolution
classifier.add(Conv2D(32, (5, 5), input_shape = (32, 32, 3), activation = 'relu')) # 32 feature detectors and 32 feature maps

# Step2 : Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding 2nd covolution layer
classifier.add(Convolution2D(32, (5,5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step3 : Flattening 
classifier.add(Flatten())

# Step4 : Full Connection
classifier.add(Dense(units = 1000, activation = 'relu')) # Hidden Layer
classifier.add(Dense(units = 10, activation = 'softmax')) # Output Layer


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the model
hist = classifier.fit(X_train, y_train_one_hot, batch_size = 265, epochs = 10, validation_split= 0.3)


classifier.evaluate(X_test, y_test_one_hot)

# Visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

# Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()

# Load the image data
my_img = plt.imread('cat.jpg')
img = plt.imshow(my_img)

# Resize the image
from skimage.transform import resize
my_image_resized = resize(my_img, (32,32,3))
img = plt.imshow(my_image_resized)

# Get the probabilities of each class
import numpy as np
probabilities = classifier.predict(np.array([my_image_resized,]))

# Print the probabilities
print(probabilities)

number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0, :])
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])
print('Second likely class:', number_to_class[index[8]], '--probability:', probabilities[0, index[8]])
print('Third likely class:', number_to_class[index[7]], '--probability:', probabilities[0, index[7]])
print('Fourth likely class:', number_to_class[index[6]], '--probability:', probabilities[0, index[6]])
print('Fifth likely class:', number_to_class[index[5]], '--probability:', probabilities[0, index[5]])


