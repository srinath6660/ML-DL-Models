import tensorflow as tf
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test  = tf.keras.utils.normalize(X_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 3)

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss, val_acc)

import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap = plt.cm.binary)
plt.show()


'''model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader')
predictions = new_model.predict(X_test)
print(predictions)

import numpy as np
print(np.argmax(predictions[0]))'''