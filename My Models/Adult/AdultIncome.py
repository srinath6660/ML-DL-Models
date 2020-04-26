import numpy as np
import pandas as pd

dataset = pd.read_csv('adult.csv')
dataset.describe()
dataset.info()
dataset.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in dataset.columns:
    if dataset[col].dtypes == 'object':
        dataset[col] = le.fit_transform(dataset[col])
        
X = dataset.drop('income', axis=1)
y = dataset['income']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = []
result = []
names = []
model.append(('LR', LogisticRegression()))
model.append(('KNN', KNeighborsClassifier()))
model.append(('SVM', SVC()))
model.append(('NB', GaussianNB()))
model.append(('DT', DecisionTreeClassifier()))
model.append(('RF', RandomForestClassifier()))

from sklearn import model_selection
for name, models in model:
    kfold = model_selection.KFold(n_splits = 10, random_state = 0, shuffle=True)
    cv_result = model_selection.cross_val_score(models, X_train, y_train, 
                                                cv = kfold, scoring = 'accuracy')
    result.append(cv_result)
    names.append(name)
    msg = '%s,%f(%f)'%(name, cv_result.mean(), cv_result.std())
    print(msg)


'''
LR,0.825421(0.007075)
KNN,0.829543(0.005850)
SVM,0.850618(0.006100)
NB,0.802681(0.005097)
DT,0.810843(0.003977)
RF,0.856624(0.004780)
'''



    
    
