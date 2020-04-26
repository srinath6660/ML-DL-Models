from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

X = mnist['data']
y = mnist['target']

import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

some_digit = X[36001]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation='nearest')

X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]

import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train==2)
y_test_2 = (y_test==2)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train_2)

classifier.predict([some_digit])

from sklearn.model_selection import cross_val_score
cvs = cross_val_score(classifier, X_train, y_train_2, scoring = 'accuracy', cv = 10)
cvs.mean()

