from sklearn.datasets import fetch_openml
mnist_data = fetch_openml('mnist_784', version=1)

print(mnist_data.keys())

X = mnist_data['data']
y = mnist_data['target']

import matplotlib.pyplot as plt

digit = X[0]
digit_pixels = digit.reshape(28, 28)
plt.subplot(131)
plt.imshow(digit_pixels)
plt.axis('off')

digit = X[1]
digit_pixels = digit.reshape(28, 28)
plt.subplot(132)
plt.imshow(digit_pixels)
plt.axis('off')

digit = X[2]
digit_pixels = digit.reshape(28, 28)
plt.subplot(133)
plt.imshow(digit_pixels)
plt.axis('off')

# Changing the labels from string to integers
import numpy as np
y = y.astype(np.uint8)


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
print('Train Data: ', X_train, '\n', 'Test Data:', X_test, '\n',
     'Train label: ', y_train, '\n', 'Test Label: ', y_test)

from sklearn.linear_model import SGDClassifier
 
sgd_clf = SGDClassifier(loss='hinge', random_state=42)
sgd_clf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')

score = sgd_clf.score(X_test, y_test)

