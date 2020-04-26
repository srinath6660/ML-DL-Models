import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('parkinsons.data')

X = dataset.drop(['name'], 1)
X = np.array(X.drop(['status'], 1))
y = np.array(dataset['status'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the XGBoost to the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Making the predictions and evaluating the model
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
 

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10 )
accuracies.mean()
accuracies.std()


from sklearn.metrics import classification_report
a = print(classification_report(y_test, y_pred))