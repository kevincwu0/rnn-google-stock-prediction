# Recurrent Neural Network
# Part 1 - Data preprocessing

# CS 229 Stanford 
# http://cs229.stanford.edu/proj2012/BernalFokPidaparthi-FinancialMarketTimeSeriesPredictionwithRecurrentNeural.pdf
# train LSTM on five years of Google 
# Supervised Deep Learning

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set = pd.read_csv("Google_Stock_Price_Train.csv")

# input/ouputs of recurrent neural (input != date, stock price)
# but stock price at time t for input, and stock price t+1 for the output
# create a set only with the "Open" Google stock price, extract that column
# two-dimensional numpy array
training_set = training_set.iloc[:,1:2].values

# Feature Scaling + Normalization, since LSTM Several Sigmoid Activation function
# Sigmoid 0 and 1, as is the case in Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler() # default is 0,1
# Fitting to training_set, scale training set, 
# transform we'll apply normalizationjust need min and max for normalization
training_set = sc.fit_transform(training_set)

# Getting the inputs and the outputs, y_train is output, x_train is the input
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshaping inputs, input has a certain format (2D array, features)
# Changing the format of X_train into a 3D array, with a timestep
# Keras Documentation - why reshape? - 3D tensor with shape (batch_size, timesteps)
# time steps different between output and input time, input_dim dimension of input feature
X_train = np.reshape(X_train, (1257, 1, 1))

# Part 2 - Building the RNN

# Importing the keras libs and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
# predicting a continuous outcome, regression model
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
# use the mean square error
# regression won't be binary cross entropy, MSE for regression
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Trainign set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
test_set = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = test_set.iloc[:,1:2].values

# Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Time Price')
plt.legend()
plt.show()

# Gettubg the real stock price 2012 - 2016
# Importing the training set
real_stock_price = pd.read_csv("Google_Stock_Price_Train.csv")
real_stock_price = real_stock_price.iloc[:,1:2].values

# Getting the predicted stock price of 2012 - 2016
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

# visualising the results
plt.plot(real_stock_price_train, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price_train, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Time Price')
plt.legend()
plt.show()

