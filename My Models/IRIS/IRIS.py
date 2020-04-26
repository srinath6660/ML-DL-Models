import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Iris.csv')


spec = np.unique(dataset['Species'])

from sklearn.preprocessing import LabelEncoder
obj = LabelEncoder()
labels = obj.fit_transform(dataset['Species'])
mappings = {index: label for index, label in 
                  enumerate(obj.classes_)}

dataset['SpeciesNumber'] = labels
X = dataset.iloc[:, 1:5]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
