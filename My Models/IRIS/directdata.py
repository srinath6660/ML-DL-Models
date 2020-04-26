from sklearn.datasets import load_iris
dataset = load_iris()

print("Target names: {}".format(dataset['target_names']))
print("Feature names: {}".format(dataset['feature_names']))
print("Type of data: {}".format(type(dataset['data'])))
print("Shape of data: {}".format(dataset['data'].shape))

print("Type of target: {}".format(type(dataset['target'])))
print("Shape of target: {}".format(dataset['target'].shape))
print("Target:\n{}".format(dataset['target']))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size = 0.25, random_state = 0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

# Same for the test samples
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

