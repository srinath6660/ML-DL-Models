import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('housing.csv')

OceanProximity = pd.get_dummies(dataset['ocean_proximity'], drop_first = True)
dataset = pd.concat([dataset, OceanProximity], axis = 1)
dataset.drop('ocean_proximity', axis = 1, inplace = True)

X = dataset.drop('median_house_value', axis = 1)
y = dataset['median_house_value']
X['total_bedrooms'].fillna(X.total_bedrooms.median(), inplace = True)

dataset.info()

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('std_scalar', StandardScaler()),('poly_features', PolynomialFeatures(degree = 2))])

housing_prepared = pipeline.fit_transform(X)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(housing_prepared, y)
regressor.score(housing_prepared, y)
p = regressor.predict(housing_prepared[:10])
