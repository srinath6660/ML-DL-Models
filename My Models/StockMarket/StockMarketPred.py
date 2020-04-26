# This program predicts stock prices
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Get the stock data
dataset = quandl.get('WIKI/FB')

# Variable for predicting 'n' days out into the future
forecast_out = 30
dataset['Prediction'] = dataset[['Adj. Close']].shift(-forecast_out)

# Convert the dataframe to numpy array
X = np.array(dataset.drop(['Prediction'], 1))
# Remove the last 'n' rows 
X = X[:-forecast_out]
y = np.array(dataset['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

regressor_svm = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)
regressor_svm.fit(X_train, y_train)

svm_confidence = regressor_svm.score(X_test, y_test)

regressor_lin = LinearRegression()
regressor_lin.fit(X_train, X_test)