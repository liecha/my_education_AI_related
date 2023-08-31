# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 04:55:10 2021

@author: Emelie Chandni
"""

# ----- PART 1 -----
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import training set of data
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# Select data of interest --> save this data in a numpy array
# (Use .values to convert array to an numpy array)
training_set = dataset_train.iloc[:, 1:2].values

# Scaling the training_set (use standardisation or normalization)
# Since our output signal is continous (sigmoid function)
# --> NORMALIZATION is recommended
from sklearn.preprocessing import MinMaxScaler

# Implement a scaler with value (range) between 0 and 1
scaler = MinMaxScaler(feature_range = (0,1))

# Apply scaler in training set
training_set_scaled = scaler.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output
# timesteps --> 60 previous days are used to predict the next output
# X_train --> 60 previous Stock Price Values
# y_traing --> value at the current day
# NOTE: [] --> type list NOT np array
X_train = []
y_train = [] 

timestep = 60
len_trainset = len(dataset_train) # len 1258

for i in range(timestep, len_trainset):
    X_train.append(training_set_scaled[i-timestep:i,0])
    y_train.append(training_set_scaled[i,0])

# Convert the lists X_train and y_train to np.arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape data to fulfill requirement of inputs for the RNN 
# RNN == (Recurrent Neural Network)
# We currently have an 2D matrix - X_train --> (1196, 60) matrix
# We want 3D matrix - the third dimension corresponds to the indicator
# In this case the indicator is 1 since we have one output
# array.shape[0] --> antal rader
# array.shape[1] --> antal kolumner
# reshape(arrayLike, newShape, order)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# len X_train 1198, y_Train 1998
print('X_train len_trainset - timestep: ', len_trainset - timestep)
print('X_train.shape ', X_train.shape)
print('y_train.shape ', y_train.shape)

# ----- PART 2 -----
# Buildning the RNN (Recurrent Neural Network)
# Import Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN 
# The output is continous value --> use regression
regressor = Sequential()

# Add the first layer to the neural network
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

# Dropout regulation (use to be 20%)
# --> this is the number of neurons to be ignored
regressor.add(Dropout(0.2))

# Add more layers to the neural network
# Second layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Third layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Forth layer - last layer before output  layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
# NOTE: Optimizer - RMSprops is recommended for RNN but Adam was detected to 
# be a better choice for this problem
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training - fit the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# ----- PART 3 -----
# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
len_testset = len(dataset_test)
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
test = pd.concat((dataset_train, dataset_test), axis = 0)
print('--- CONCAT TESTSET ---')
print(test)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(timestep, timestep+len_testset):
    X_test.append(inputs[i-timestep:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# len X_test 20
print('X_test (len_testset, timestep, 1) == (20, 60, 1): ', len_testset)
print('X_test.shape ', X_test.shape)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

date = dataset_test['Date']

# Visualising the results
plt.plot(date, real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(date, predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()




































