import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('cardio_train.csv', sep = ';')


dataset.isnull().sum()

dataset['cardio'].value_counts()

sns.countplot(dataset['cardio'])

dataset['years'] = (dataset['age']/365).round(0)
dataset['years'] = pd.to_numeric(dataset['years'], downcast = 'integer')    
sns.countplot(x = 'years', hue = 'cardio', data = dataset, palette = 'colorblind', 
              edgecolor = sns.color_palette('dark', n_colors = 1))                

dataset.corr()

plt.figure(figsize = (7,7))
sns.heatmap(dataset.corr(), annot = True, fmt = '.0%')

dataset = dataset.drop('years', axis = 1)
dataset = dataset.drop('id', axis = 1)

X = dataset.iloc[:, :12].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25 , random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', 
                                    random_state = 1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

classifier.score(X_train, y_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)